#!/usr/bin/env python3
"""vllm-model-archive — move HF checkpoint caches between internal storage and an
archive disk, keeping the model profiles (flags/config) in place.

A checkpoint has to be present on *both* sparks before vLLM can serve it (the
pair is tensor-parallel), so archive/unarchive act on every node: each node
moves its own copy to its own archive disk, and every size is reported per node.

Layout (per node):

    $ARCHIVE_PATH/models/<safe-name>/
        manifest.json          what/when/how big, for that node
        hub/                   cache-root-relative tree that was archived
            models--org--name/...
            blobs/xx/<sha>     external blob targets, only when referenced

Nothing outside the HF cache is touched: flags profiles
(`~/.config/spark-vllm/flags/*`), `model.conf` and `locked.conf` stay as they are,
which is what makes an archived model still listed, lockable and re-selectable.

CLI (add --json for machine-readable output):

    vllm-model-archive --status                     # archive config, per node
    vllm-model-archive --set-path /archive          # configure *this* node's path
    vllm-model-archive --list                        # models + per-node state
    vllm-model-archive --archive <model-id> [--dry-run] [--node all|head|worker]
    vllm-model-archive --unarchive <model-id> [--dry-run] [--prune-archive]

Swap safety: the copy phase runs on every node first and deletes nothing; the
internal copies are removed only after every node's copy verifies.  A failure
during the copy phase rolls the staged archive dirs back, so the cluster is
never left half-archived.

Exit codes: 0 ok · 2 refused/precondition · 3 does not fit · 4 I/O failure ·
5 a node is unreachable or its tool is missing
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

TOOL = "vllm-model-archive 0.2.0"
CACHE_ROOT = os.path.expanduser("~/.cache/huggingface/hub")
CONF_DIR = os.path.expanduser("~/.config/spark-vllm")
ARCHIVE_CONF = os.path.join(CONF_DIR, "archive.conf")
MODEL_CONF = os.path.join(CONF_DIR, "model.conf")
FLAGS_DIR = os.path.join(CONF_DIR, "flags")
DEFAULT_MIN_FREE_GB = 10.0
# offered as the default in the path prompt when it is a real mount; being
# mounted does NOT mean configured - --set-path is still required per node
DEFAULT_ARCHIVE = os.environ.get("SPARK_VLLM_DEFAULT_ARCHIVE", "/archive")


# ---------------------------------------------------------------- config files

def read_kv(path):
    d = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    d[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return d


def write_kv(path, data, header):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w") as f:
        f.write(header)
        for k, v in data.items():
            f.write(f"{k}={v}\n")
    os.replace(tmp, path)


def archive_conf():
    d = read_kv(ARCHIVE_CONF)
    return {
        "path": d.get("ARCHIVE_PATH", ""),
        "min_free_gb": float(d.get("MIN_FREE_GB", DEFAULT_MIN_FREE_GB) or DEFAULT_MIN_FREE_GB),
    }


def save_archive_conf(path, min_free_gb=DEFAULT_MIN_FREE_GB):
    write_kv(ARCHIVE_CONF, {"ARCHIVE_PATH": path, "MIN_FREE_GB": min_free_gb},
             "# archive disk for model checkpoints (managed by vllm-model-archive)\n"
             "# ARCHIVE_PATH is mounted by the operator; MIN_FREE_GB is the safety\n"
             "# margin kept free on the internal cache filesystem.\n")


def boot_model():
    return read_kv(MODEL_CONF).get("MODEL_ID", "")


# ---------------------------------------------------------------------- nodes

NODE_ORDER = os.environ.get("SPARK_VLLM_NODES", "head,worker")
LOCAL_NODE = os.environ.get("SPARK_VLLM_NODE", "head")
WORKER_HOST = os.environ.get("WORKER", "gx10-02")
HELPER = os.environ.get("SPARK_VLLM_ARCHIVE_HELPER", "~/.local/bin/vllm-model-archive.py")
REMOTE_CMD = os.environ.get(
    "SPARK_VLLM_REMOTE_CMD",
    "ssh -o BatchMode=yes -o ConnectTimeout=10 {host} {helper} {args}")


def node_names(scope="all"):
    ns = [n.strip() for n in NODE_ORDER.split(",") if n.strip()]
    if scope in ("", "all", None):
        return ns
    if scope not in ns:
        raise ValueError(f"unknown node {scope!r} (known: {', '.join(ns)})")
    return [scope]


def set_local_node(name):
    """Tell this process which node it is (used for the ssh side)."""
    global LOCAL_NODE
    LOCAL_NODE = name


def node_host(node):
    return "" if node == LOCAL_NODE else WORKER_HOST


def _json_from(text):
    """Parse the helper's JSON out of possibly-noisy login-shell output."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except ValueError:
                return None
        return None


def call_node(node, args, timeout=7200):
    """Run this same CLI on <node> and return (code, result-dict)."""
    if node == LOCAL_NODE:
        code, res = dispatch(list(args))
        res = dict(res)
        res.setdefault("node", node)
        res["reachable"] = True
        return code, res
    import shlex
    argv = list(args) + ([] if "--json" in args else ["--json"])
    if "--self-node" not in argv:
        argv += ["--self-node", node]
    cmd = REMOTE_CMD.format(host=node_host(node), helper=HELPER,
                            args=" ".join(shlex.quote(a) for a in argv))
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 5, {"ok": False, "node": node, "reachable": False,
                   "error": f"timed out after {timeout}s"}
    except OSError as e:
        return 5, {"ok": False, "node": node, "reachable": False,
                   "error": f"cannot run the remote command: {e}"}
    res = _json_from(r.stdout)
    if res is None:
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        return 5, {"ok": False, "node": node, "reachable": False,
                   "error": "the archive tool is not usable there",
                   "detail": detail[-1][:200] if detail else "no output",
                   "hint": "deploy.sh --all installs it on the worker"}
    res["node"] = node
    res["reachable"] = True
    return r.returncode, res


# ------------------------------------------------------------------ semantics

def engine_for(mid):
    """Same dispatch rule as spark-vllm-podman-start-select."""
    return "miaai" if "Qwen3.8-Flash-Next-NVFP4" in mid else "b12x"


def safe_name(mid):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", mid.replace("/", "-"))[:60]


def repo_rel(mid):
    return "models--" + mid.replace("/", "--")


def repo_dir(mid):
    return os.path.join(CACHE_ROOT, repo_rel(mid))


def archive_root():
    return archive_conf()["path"]


def model_archive_dir(mid, path=None):
    path = path or archive_root()
    return os.path.join(path, "models", safe_name(mid))


def profile_files(mid):
    base = os.path.join(FLAGS_DIR, safe_name(mid))
    return [p for p in (base + ".conf", base + ".env") if os.path.exists(p)]


def du_bytes(path):
    """Apparent size of a tree, hardlinks counted once, symlinks not followed."""
    total, seen = 0, set()
    for root, dirs, names in os.walk(path, followlinks=False):
        for n in names + dirs:
            p = os.path.join(root, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p):
                total += st.st_size
                continue
            if not os.path.isfile(p):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    return total


def payload(mid, cache_root=CACHE_ROOT):
    """Files to move for one model: its repo dir plus every external blob the
    repo's symlinks resolve to (Xet stores weights under <cache>/blobs/xx/<sha>).

    Returns dict(entries=[relpaths], externals=[relpaths], bytes, files).
    """
    rd = os.path.join(cache_root, repo_rel(mid))
    entries, externals, dangling = [], [], []
    seen, total, files = set(), 0, 0
    for root, dirs, names in os.walk(rd, followlinks=False):
        for n in names:
            p = os.path.join(root, n)
            entries.append(os.path.relpath(p, cache_root))
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p):
                total += st.st_size
                tgt = os.path.realpath(p)
                if not os.path.exists(tgt):
                    dangling.append(os.path.relpath(p, cache_root))
                    continue
                # Inside the repo dir: covered by the walk. Elsewhere: the real
                # bytes live in the shared blob store and must travel with it.
                if not (tgt == rd or tgt.startswith(rd + os.sep)) and tgt.startswith(cache_root + os.sep):
                    rel = os.path.relpath(tgt, cache_root)
                    if rel not in externals:
                        externals.append(rel)
                continue
            if not os.path.isfile(p):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
            files += 1
    for rel in externals:
        p = os.path.join(cache_root, rel)
        try:
            st = os.stat(p)
        except OSError:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += st.st_size
        files += 1
    return {"entries": entries, "externals": externals, "bytes": total,
            "files": files, "dangling": dangling}


def cached_refs(cache_root=CACHE_ROOT, exclude_mid=None):
    """Realpaths referenced by snapshots of every cached model except one."""
    refs = set()
    skip = os.path.join(cache_root, repo_rel(exclude_mid)) if exclude_mid else None
    for d in os.listdir(cache_root) if os.path.isdir(cache_root) else []:
        if not d.startswith("models--"):
            continue
        rd = os.path.join(cache_root, d)
        if skip and rd == skip:
            continue
        snaps = os.path.join(rd, "snapshots")
        if not os.path.isdir(snaps):
            continue
        for root, _dirs, names in os.walk(snaps):
            for n in names:
                p = os.path.join(root, n)
                if os.path.islink(p):
                    refs.add(os.path.realpath(p))
    return refs


def free_estimate(mid, pay=None):
    """Bytes archiving <mid> would actually free here: its payload minus the
    blobs another cached model still points at (the two Qwen twins share ~57 GB
    on the head, so its nominal payload overstates the win)."""
    pay = pay or payload(mid)
    refs = cached_refs(exclude_mid=mid)
    shared = 0
    for rel in pay["externals"]:
        p = os.path.join(CACHE_ROOT, rel)
        try:
            if os.path.realpath(p) in refs:
                shared += os.path.getsize(p)
        except OSError:
            continue
    return max(0, pay["bytes"] - shared)


def manifest_bytes(man):
    """Payload size from a manifest, v2 (`payload`) or legacy (`nodes.head`)."""
    if isinstance(man.get("payload"), dict):
        return int(man["payload"].get("bytes") or 0)
    return int((man.get("nodes", {}).get("head", {}) or {}).get("payload_bytes") or 0)


def scan_archived(path=None):
    out = {}
    root = os.path.join(path or archive_root() or "", "models")
    if not root or not os.path.isdir(root):
        return out
    for d in sorted(os.listdir(root)):
        mf = os.path.join(root, d, "manifest.json")
        try:
            with open(mf) as f:
                m = json.load(f)
            out[m["model_id"]] = m
        except (OSError, ValueError, KeyError):
            continue
    return out


def models_local(path=None):
    """Models this node knows: cached, archived, or half-moved (staged)."""
    arch = scan_archived(path)
    out = {}
    if os.path.isdir(CACHE_ROOT):
        for d in sorted(os.listdir(CACHE_ROOT)):
            if not d.startswith("models--") or "--" not in d[len("models--"):]:
                continue
            org, mod = d[len("models--"):].split("--", 1)
            mid = f"{org}/{mod}"
            out[mid] = {"id": mid, "engine": engine_for(mid), "cached": True,
                        "archived": mid in arch, "profiles": profile_files(mid),
                        # payload(), not du: Xet-backed weights sit outside the
                        # repo dir, so a plain du understates the model by ~10x
                        "bytes": payload(mid)["bytes"]}
    for mid, m in arch.items():
        row = out.setdefault(mid, {"id": mid, "engine": m.get("engine", engine_for(mid)),
                                   "cached": False, "profiles": profile_files(mid),
                                   "bytes": 0})
        row["archived"] = True
        row["mstate"] = m.get("state", "archived")
        row["archive_bytes"] = manifest_bytes(m)
        row["archived_at"] = m.get("created_at", "")
    for row in out.values():
        row.setdefault("archived", False)
        row["state"] = ("staged" if row.get("mstate") == "staged"
                        else "cached+archived" if row["cached"] and row["archived"]
                        else "cached" if row["cached"] else "archived")
        row["archive_bytes"] = row.get("archive_bytes", 0)
    return [out[k] for k in sorted(out)]


def merge_nodes(per_node):
    """Merge per-node model lists into cluster rows carrying per-node detail."""
    rows = {}
    for node, res in per_node.items():
        if not res or not res.get("ok"):
            continue
        for m in res.get("models", []):
            row = rows.setdefault(m["id"], {"id": m["id"], "engine": m["engine"],
                                            "nodes": {}, "profiles": m.get("profiles", [])})
            if isinstance(m.get("nodes"), dict):
                # a node that already answered with its own merged rows
                row["nodes"].update(m["nodes"])
            else:
                row["nodes"][node] = {"cached": bool(m.get("cached")),
                                      "archived": bool(m.get("archived")),
                                      "state": m.get("state", ""),
                                      "bytes": int(m.get("bytes") or 0),
                                      "archive_bytes": int(m.get("archive_bytes") or 0)}
            row["profiles"] = row["profiles"] or m.get("profiles", [])
    boot, live = boot_model(), served_model()
    for row in rows.values():
        states = [n.get("state") for n in row["nodes"].values()]
        cached = [n["cached"] for n in row["nodes"].values()]
        row["cached"] = bool(cached) and all(cached)
        row["archived"] = bool(cached) and all(n["archived"] for n in row["nodes"].values())
        row["state"] = ("cached" if row["cached"] else
                        "archived" if row["archived"] else
                        "staged" if "staged" in states else "partial")
        row["cache_bytes"] = sum(n["bytes"] for n in row["nodes"].values())
        row["archive_bytes"] = sum(n["archive_bytes"] for n in row["nodes"].values())
        row["current"] = row["id"] == boot
        row["serving"] = bool(live) and row["id"] == live
        row["active"] = row["current"] or row["serving"]
    return [rows[k] for k in sorted(rows)]


def fit_report(mid, path=None):
    """Can <mid> be pulled back into this node's cache right now?"""
    conf = archive_conf()
    adir = model_archive_dir(mid, path)
    mf = os.path.join(adir, "manifest.json")
    if not os.path.exists(mf):
        return {"fits": False, "reason": "not archived", "needed_bytes": 0,
                "free_bytes": 0, "suggestions": []}
    with open(mf) as f:
        man = json.load(f)
    hub = os.path.join(adir, "hub")
    needed = 0
    for root, _dirs, names in os.walk(hub):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, hub)
            dest = os.path.join(CACHE_ROOT, rel)
            try:
                if os.path.islink(p):
                    continue
                size = os.path.getsize(p)
            except OSError:
                continue
            if os.path.exists(dest):
                try:
                    if os.path.getsize(dest) == size:
                        continue
                except OSError:
                    pass
            needed += size
    free = shutil.disk_usage(CACHE_ROOT).free
    margin = int(conf["min_free_gb"] * 1e9)
    fits = free - margin >= needed
    suggestions = []
    sufficient = True
    if not fits:
        shortfall = needed + margin - free
        cands = [m for m in models_local(path)
                 if m["cached"] and m["id"] != mid and m["id"] != boot_model()]
        cands.sort(key=lambda m: m["bytes"], reverse=True)
        acc, picked = 0, []
        for m in cands:
            picked.append(m)
            acc += free_estimate(m["id"])
            if acc >= shortfall:
                break
        # Best effort even when no combination is enough: the user still needs
        # to see what the biggest wins would be.
        sufficient = acc >= shortfall
        suggestions = [{"id": m["id"], "frees_bytes": free_estimate(m["id"])}
                       for m in picked]
    return {"fits": fits, "needed_bytes": needed, "free_bytes": free,
            "margin_bytes": margin, "fits_have_bytes": free - margin,
            "shortfall_bytes": max(0, needed + margin - free),
            "suggestions_sufficient": sufficient,
            "suggestions": suggestions,
            "archive_bytes": manifest_bytes(man),
            "archived_at": man.get("created_at", "")}


# --------------------------------------------------------------- file moving

def rsync(src_root, dst_root, rels, dry_run=False):
    """Copy the listed cache-root-relative paths, preserving symlinks."""
    import tempfile
    fd, listfile = tempfile.mkstemp(prefix="vllm-archive-", suffix=".list")
    try:
        with os.fdopen(fd, "wb") as f:
            for r in rels:
                f.write(r.encode() + b"\0")
        cmd = ["rsync", "-aH", "--numeric-ids", "--from0",
               "--files-from=" + listfile]
        if dry_run:
            cmd.append("-n")
        cmd += [src_root.rstrip("/") + "/", dst_root.rstrip("/") + "/"]
        return subprocess.run(cmd, capture_output=True, text=True)
    finally:
        try:
            os.unlink(listfile)
        except OSError:
            pass


def copy_paths(src_root, dst_root, rels):
    """Copy cache-root-relative paths to another root, preserving symlinks.

    rsync when it is installed (it is on the DGX images), otherwise an internal
    copy so the tool still works on a minimal host.  Existing destination files
    of the same size are left alone, which makes an interrupted copy resumable.
    Returns (method, files_copied, error).
    """
    if shutil.which("rsync"):
        r = rsync(src_root, dst_root, rels)
        if r.returncode == 0:
            return "rsync", len(rels), ""
        return "rsync", 0, (r.stderr or r.stdout).strip()[:300]

    copied, errors = 0, []
    for rel in rels:
        s = os.path.join(src_root, rel)
        d = os.path.join(dst_root, rel)
        try:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            if os.path.islink(s):
                target = os.readlink(s)
                if os.path.islink(d) and os.readlink(d) == target:
                    continue
                if os.path.lexists(d):
                    os.unlink(d)
                os.symlink(target, d)
                continue
            if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
                continue
            shutil.copy2(s, d)
            copied += 1
        except OSError as e:
            errors.append(f"{rel}: {e}")
            if len(errors) >= 5:
                break
    return "copy", copied, "; ".join(errors)


def verify_tree(src_root, dst_root, rels):
    """File-by-file check after a copy: every entry exists with the same type
    and size, symlinks keep their target string.

    Deliberately not `rsync --delete`: with --files-from that can remove
    unlisted files in shared directories (e.g. the blob store), and the
    destination for a restore is the live cache root.
    """
    bad = []
    for rel in rels:
        s, d = os.path.join(src_root, rel), os.path.join(dst_root, rel)
        try:
            if os.path.islink(s):
                if not (os.path.islink(d) and os.readlink(s) == os.readlink(d)):
                    bad.append(rel)
                continue
            ss, ds = os.stat(s), os.stat(d)
            if ss.st_size != ds.st_size:
                bad.append(rel)
        except OSError:
            bad.append(rel)
        if len(bad) >= 20:
            break
    return bad


def prune_empty_dirs(path, stop):
    path = os.path.dirname(path)
    stop = stop.rstrip("/")
    while path.startswith(stop) and path != stop:
        try:
            os.rmdir(path)
        except OSError:
            return
        path = os.path.dirname(path)


# ----------------------------------------------------------------- operations

def human(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000.0
    return f"{n:.1f} TB"


def served_model():
    """Model id the live engine answers with, '' when it is not serving."""
    import urllib.request
    url = os.environ.get("SPARK_VLLM_ENGINE_URL", "http://127.0.0.1:8000").rstrip("/")
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=2) as r:
            return json.load(r)["data"][0]["id"]
    except Exception:
        return ""


def service_active():
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-p", "ActiveState",
                              "--value", "spark-vllm-podman.service"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"
    return out


def detected_archive_path():
    """The conventional archive mount, if it exists here as a separate fs."""
    cand = os.path.abspath(os.path.expanduser(DEFAULT_ARCHIVE))
    try:
        if (os.path.isdir(cand) and os.path.ismount(cand)
                and os.stat(cand).st_dev != os.stat(CACHE_ROOT).st_dev
                and os.access(cand, os.W_OK)):
            return cand
    except OSError:
        pass
    return ""


def op_set_path(path):
    if not path:
        return 2, {"ok": False, "error": "empty path"}
    ap = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(ap):
        return 2, {"ok": False, "error": f"{ap} does not exist (create and mount the disk first)"}
    if not os.access(ap, os.W_OK):
        return 2, {"ok": False, "error": f"{ap} is not writable"}
    try:
        cfs = os.stat(CACHE_ROOT).st_dev
        afs = os.stat(ap).st_dev
    except OSError as e:
        return 2, {"ok": False, "error": str(e)}
    if cfs == afs:
        return 2, {"ok": False,
                   "error": f"{ap} is on the same filesystem as the cache - archiving would not free space"}
    save_archive_conf(ap, archive_conf()["min_free_gb"])
    disk = shutil.disk_usage(ap)
    return 0, {"ok": True, "path": ap, "free_bytes": disk.free, "total_bytes": disk.total}


def _write_manifest(adir, man):
    with open(os.path.join(adir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)


def _read_manifest(adir):
    try:
        with open(os.path.join(adir, "manifest.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def op_set_path_cluster(path, scope="all"):
    """Write archive.conf on one or more nodes (the head drives the worker)."""
    try:
        nodes = node_names(scope)
    except ValueError as e:
        return 2, {"ok": False, "error": str(e)}
    per = {}
    for n in nodes:
        _c, per[n] = call_node(n, ["--set-path", path, "--local"], timeout=180)
    bad = [n for n, r in per.items() if not r.get("ok")]
    if bad:
        return 2, {"ok": False, "path": path, "nodes": per, "failed_nodes": bad,
                   "error": "; ".join(f"{n}: {(per[n].get('error') or 'failed')}" for n in bad)}
    return 0, {"ok": True, "path": path, "nodes": per}


def archive_local(mid, stage="commit", dry_run=False):
    """One node's half of an archive.

    plan      validate + report sizes (no writes)
    copy      write manifest state=staged, copy, verify (deletes nothing)
    commit    delete the internal copy + unshared blobs, mark archived
    rollback  drop a staged archive dir (only while the source is intact)
    """
    conf = archive_conf()
    rd = repo_dir(mid)
    adir = model_archive_dir(mid)
    man = _read_manifest(adir)

    if stage == "rollback":
        if not man or man.get("state") != "staged":
            return 0, {"ok": True, "model_id": mid, "rolled_back": False,
                       "note": "nothing staged here"}
        if not os.path.isdir(rd):
            return 2, {"ok": False, "error": "internal copy is gone - keeping the archive dir"}
        shutil.rmtree(adir, ignore_errors=True)
        return 0, {"ok": True, "model_id": mid, "rolled_back": True}

    if not os.path.isdir(rd):
        if man:
            if man.get("state") == "staged":
                # a previous run copied and verified, then died mid-delete:
                # finish the bookkeeping instead of failing forever
                man["state"] = "archived"
                man["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                man.setdefault("nodes", {}).setdefault(LOCAL_NODE, {})["freed_bytes"] = 0
                _write_manifest(adir, man)
                return 0, {"ok": True, "model_id": mid, "finalized": True,
                           "freed_bytes": 0, "bytes": manifest_bytes(man),
                           "note": "source already gone; archive copy kept"}
            return 2, {"ok": False, "error": f"{mid} is already archived"}
        return 2, {"ok": False, "error": f"{mid} is not in the cache ({rd})"}

    # the active model cannot be archived: its weights must stay in the cache
    # (boot default would break the next start; a served model is in use now)
    if stage in ("plan", "copy") or dry_run:
        if mid == boot_model():
            return 2, {"ok": False, "model_id": mid, "reason": "boot default",
                       "error": (f"cannot archive the current model ({mid}): it is the boot "
                                 f"default in model.conf - switch to another model/engine first")}
        live = served_model()
        if live and live == mid:
            return 2, {"ok": False, "model_id": mid, "reason": "serving",
                       "service": service_active(),
                       "error": (f"cannot archive the current model ({mid}): the engine is "
                                 f"serving it right now - switch to another model/engine first")}

    if not conf["path"]:
        return 2, {"ok": False, "error": "archive path not configured here (run --set-path /archive)"}
    if not os.path.isdir(conf["path"]):
        return 2, {"ok": False, "error": f"archive path {conf['path']} is not mounted here"}

    pay = payload(mid)
    if not pay["entries"]:
        return 2, {"ok": False, "error": f"nothing to archive under {rd}"}
    disk = shutil.disk_usage(conf["path"])
    freed_est = free_estimate(mid, pay)
    fits = disk.free >= pay["bytes"]

    if stage == "plan" or dry_run:
        return 0, {"ok": True, "plan": True, "model_id": mid, "engine": engine_for(mid),
                   "bytes": pay["bytes"], "files": pay["files"],
                   "externals": len(pay["externals"]), "dangling": pay["dangling"],
                   "freed_est_bytes": freed_est, "fits": fits,
                   "archive_dir": adir, "archive_path": conf["path"],
                   "archive_free_bytes": disk.free,
                   "cache_free_bytes": shutil.disk_usage(CACHE_ROOT).free}

    if not fits:
        return 3, {"ok": False, "error": "archive disk full", "needed_bytes": pay["bytes"],
                   "free_bytes": disk.free, "archive_dir": adir}

    rels = pay["entries"] + pay["externals"]
    if not man or man.get("state") != "staged":
        man = {
            "schema": 2, "tool": TOOL, "model_id": mid, "engine": engine_for(mid),
            "state": "staged", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "node": LOCAL_NODE, "archive_path": conf["path"], "host": os.uname().nodename,
            "profiles": [os.path.basename(p) for p in profile_files(mid)],
            "payload": {"bytes": pay["bytes"], "files": pay["files"],
                        "externals": len(pay["externals"]), "dangling": pay["dangling"]},
            "nodes": {LOCAL_NODE: {"cache_root": CACHE_ROOT, "repo_rel": repo_rel(mid),
                                   "payload_bytes": pay["bytes"], "payload_files": pay["files"],
                                   "freed_est_bytes": freed_est,
                                   "externals": pay["externals"]}},
        }
        os.makedirs(adir, exist_ok=True)
        _write_manifest(adir, man)

    hub = os.path.join(adir, "hub")
    os.makedirs(hub, exist_ok=True)
    method, _ncopied, cerr = copy_paths(CACHE_ROOT, hub, rels)
    if cerr:
        return 4, {"ok": False, "error": f"{method} copy failed: {cerr}",
                   "archive_dir": adir}
    bad = verify_tree(CACHE_ROOT, hub, rels)
    if bad:
        return 4, {"ok": False, "error": "copy verification failed; source left intact",
                   "detail": ", ".join(bad[:5]), "archive_dir": adir}
    man["state"] = "staged"
    man["copied_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_manifest(adir, man)

    if stage == "copy":
        return 0, {"ok": True, "model_id": mid, "engine": engine_for(mid), "staged": True,
                   "bytes": pay["bytes"], "files": pay["files"], "copy_method": method,
                   "freed_est_bytes": freed_est, "archive_dir": adir,
                   "archive_free_bytes": shutil.disk_usage(conf["path"]).free}

    # Verified: drop the internal copy, then the blobs nothing else references.
    before = shutil.disk_usage(CACHE_ROOT).free
    freed, seen = 0, set()
    for rel in pay["entries"]:
        p = os.path.join(CACHE_ROOT, rel)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if os.path.islink(p) or os.path.isdir(p) or st.st_nlink > 1:
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:
            continue
        seen.add(key)
        freed += st.st_size
    shutil.rmtree(rd)
    refs = cached_refs(exclude_mid=mid)
    removed_ext = 0
    for rel in pay["externals"]:
        p = os.path.join(CACHE_ROOT, rel)
        try:
            st = os.lstat(p)
        except OSError:
            continue
        if os.path.realpath(p) in refs or st.st_nlink > 1:
            continue
        try:
            os.unlink(p)
            removed_ext += 1
            freed += st.st_size
            prune_empty_dirs(p, os.path.join(CACHE_ROOT, "blobs"))
        except OSError:
            pass
    after = shutil.disk_usage(CACHE_ROOT).free
    man["state"] = "archived"
    man["archived_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    node = man.setdefault("nodes", {}).setdefault(LOCAL_NODE, {})
    node.update({"freed_bytes": freed, "df_delta_bytes": max(0, after - before),
                 "removed_externals": removed_ext})
    _write_manifest(adir, man)
    return 0, {"ok": True, "model_id": mid, "engine": engine_for(mid),
               "bytes": pay["bytes"], "files": pay["files"], "copy_method": method,
               "externals_removed": removed_ext, "freed_bytes": freed,
               "df_delta_bytes": max(0, after - before),
               "free_after_bytes": shutil.disk_usage(CACHE_ROOT).free,
               "archive_dir": adir}


def unarchive_local(mid, stage="commit", dry_run=False, prune=False):
    """One node's half of an unarchive: fit check, copy back, verify, then
    (optionally) drop the archive copy once the restore is verified."""
    conf = archive_conf()
    adir = model_archive_dir(mid)
    mf = os.path.join(adir, "manifest.json")
    if not os.path.exists(mf):
        return 2, {"ok": False, "error": f"{mid} is not archived under {conf['path'] or '(unset)'}"}
    fit = fit_report(mid)
    if stage == "plan" or dry_run:
        return 0, {"ok": True, "plan": True, "model_id": mid, **fit,
                   "archive_path": conf["path"]}
    if stage == "commit" and not dry_run:
        man = _read_manifest(adir) or {}
        node_meta = man.setdefault("nodes", {}).setdefault(LOCAL_NODE, {})
        restored = int(node_meta.get("restored_bytes") or 0)
        man["state"] = "restored"
        man["restored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_manifest(adir, man)
        if prune:
            shutil.rmtree(adir, ignore_errors=True)
        return 0, {"ok": True, "model_id": mid, "model_restored": True,
                   "restored_bytes": restored or fit["needed_bytes"],
                   "pruned": bool(prune),
                   "free_after_bytes": shutil.disk_usage(CACHE_ROOT).free}
    if not fit["fits"]:
        return 3, {"ok": False, "error": "not enough internal space", **fit}
    hub = os.path.join(adir, "hub")
    rels = []
    for root, _dirs, names in os.walk(hub):
        for n in names:
            rels.append(os.path.relpath(os.path.join(root, n), hub))
    os.makedirs(CACHE_ROOT, exist_ok=True)
    method, _ncopied, cerr = copy_paths(hub, CACHE_ROOT, rels)
    if cerr:
        return 4, {"ok": False, "error": f"{method} restore failed: {cerr}"}
    bad = verify_tree(hub, CACHE_ROOT, rels)
    if bad:
        return 4, {"ok": False, "error": "restore verification failed",
                   "detail": ", ".join(bad[:5])}
    man = _read_manifest(adir) or {}
    man["state"] = "restored"
    man["restored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # record the size now: after the copy it is already back in the cache, so a
    # fresh fit check on the commit stage would report 0
    man.setdefault("nodes", {}).setdefault(LOCAL_NODE, {})["restored_bytes"] = fit["needed_bytes"]
    _write_manifest(adir, man)
    return 0, {"ok": True, "model_id": mid, "restored": True,
               "restored_bytes": fit["needed_bytes"], "files": len(rels),
               "copy_method": method, "needed_bytes": fit["needed_bytes"],
               "free_after_bytes": shutil.disk_usage(CACHE_ROOT).free}


# ---------------------------------------------------------------- orchestration

def _first_error(per_node):
    for n, res in per_node.items():
        if not res.get("ok"):
            return n, res
    return None, None


def op_archive(mid, dry_run=False, scope="all", timeout=7200):
    """Archive <mid> on every node: plan, copy+verify everywhere, then delete."""
    try:
        nodes = node_names(scope)
    except ValueError as e:
        return 2, {"ok": False, "error": str(e)}
    plans = {}
    for n in nodes:
        _c, plans[n] = call_node(n, ["--archive", mid, "--dry-run", "--local"], timeout=900)
    bad_node, bad = _first_error(plans)
    if bad_node:
        err = str(bad.get("error", ""))
        code = 3 if ("space" in err or "full" in err) else 5 if not bad.get("reachable", True) else 2
        return code, {"ok": False, "model_id": mid, "node": bad_node,
                      "error": f"{bad_node}: {bad.get('error')}", "reason": bad.get("reason"),
                      "detail": bad.get("detail"), "hint": bad.get("hint"),
                      "needed_bytes": bad.get("needed_bytes"),
                      "free_bytes": bad.get("free_bytes"), "nodes": plans}
    total_bytes = sum(int(p.get("bytes") or 0) for p in plans.values())
    total_freed = sum(int(p.get("freed_est_bytes") or 0) for p in plans.values())
    if dry_run:
        return 0, {"ok": True, "dry_run": True, "model_id": mid, "engine": engine_for(mid),
                   "nodes": plans, "total_bytes": total_bytes,
                   "total_freed_est_bytes": total_freed}

    staged = {}
    for n in nodes:
        _c, res = call_node(n, ["--archive", mid, "--stage", "copy", "--local"], timeout=timeout)
        if not res.get("ok"):
            rolled = []
            for m in staged:
                _c2, r = call_node(m, ["--archive", mid, "--stage", "rollback", "--local"], timeout=900)
                if r.get("rolled_back"):
                    rolled.append(m)
            return 4, {"ok": False, "model_id": mid, "node": n,
                       "error": f"{n}: {res.get('error')}",
                       "detail": res.get("detail"), "rolled_back": rolled,
                       "nodes": {**plans, **staged, n: res}}
        staged[n] = res

    done, failed = {}, []
    for n in nodes:
        _c, res = call_node(n, ["--archive", mid, "--stage", "commit", "--local"], timeout=timeout)
        done[n] = res
        if not res.get("ok"):
            failed.append(n)
    if failed:
        return 4, {"ok": False, "model_id": mid, "failed_nodes": failed,
                   "error": f"copied and verified everywhere, but the internal copy could not "
                            f"be dropped on {', '.join(failed)} - re-run --archive {mid}",
                   "nodes": done}
    freed = sum(int(r.get("freed_bytes") or 0) for r in done.values())
    return 0, {"ok": True, "model_id": mid, "engine": engine_for(mid), "nodes": done,
               "total_bytes": total_bytes, "total_freed_bytes": freed,
               "freed_bytes": freed, "bytes": total_bytes,
               "files": sum(int(r.get("files") or 0) for r in done.values())}


def op_unarchive(mid, dry_run=False, prune=False, scope="all", timeout=7200):
    """Restore <mid> on every node after checking each node's free space."""
    try:
        nodes = node_names(scope)
    except ValueError as e:
        return 2, {"ok": False, "error": str(e)}
    plans = {}
    for n in nodes:
        _c, plans[n] = call_node(n, ["--unarchive", mid, "--dry-run", "--local"], timeout=900)
    bad_node, bad = _first_error(plans)
    if bad_node:
        return 2, {"ok": False, "model_id": mid, "node": bad_node,
                   "error": f"{bad_node}: {bad.get('error')}", "nodes": plans}
    short = {n: p for n, p in plans.items() if not p.get("fits")}
    if short:
        return 3, {"ok": False, "model_id": mid, "error": "not enough internal space",
                   "nodes": plans, "no_space_nodes": sorted(short),
                   "shortfall_nodes": {n: p.get("shortfall_bytes", 0) for n, p in short.items()}}
    if dry_run:
        return 0, {"ok": True, "dry_run": True, "model_id": mid, "nodes": plans,
                   "total_needed_bytes": sum(int(p.get("needed_bytes") or 0) for p in plans.values())}

    copied = {}
    for n in nodes:
        _c, res = call_node(n, ["--unarchive", mid, "--stage", "copy", "--local"], timeout=timeout)
        copied[n] = res
        if not res.get("ok"):
            return 4, {"ok": False, "model_id": mid, "node": n,
                       "error": f"{n}: {res.get('error')}",
                       "note": "restores are additive; re-run to resume",
                       "nodes": {**plans, **copied}}
    done, failed = {}, []
    for n in nodes:
        args = ["--unarchive", mid, "--local"] + (["--prune-archive"] if prune else [])
        _c, res = call_node(n, args, timeout=timeout)
        done[n] = res
        if not res.get("ok"):
            failed.append(n)
    if failed:
        return 4, {"ok": False, "model_id": mid, "failed_nodes": failed,
                   "error": f"restored on {', '.join(n for n in nodes if n not in failed)} "
                            f"but not on {', '.join(failed)}", "nodes": done}
    restored = sum(int(r.get("restored_bytes") or 0) for r in done.values())
    return 0, {"ok": True, "model_id": mid, "nodes": done,
               "total_restored_bytes": restored, "restored_bytes": restored}


# ---------------------------------------------------------------------- status

def _node_status_local():
    conf = archive_conf()
    out = {"ok": True, "node": LOCAL_NODE, "configured": bool(conf["path"]),
           "archive_path": conf["path"], "min_free_gb": conf["min_free_gb"],
           "suggested_path": detected_archive_path(),
           "cache_root": CACHE_ROOT,
           "cache_free_bytes": shutil.disk_usage(CACHE_ROOT).free,
           "archived": sorted(scan_archived(conf["path"]))}
    if conf["path"] and os.path.isdir(conf["path"]):
        d = shutil.disk_usage(conf["path"])
        out.update({"mounted": True, "archive_free_bytes": d.free,
                    "archive_total_bytes": d.total})
    else:
        out.update({"mounted": False})
    return out


def op_status(scope="all"):
    try:
        nodes = node_names(scope)
    except ValueError as e:
        return 2, {"ok": False, "error": str(e)}
    per = {}
    for n in nodes:
        if n == LOCAL_NODE:
            per[n] = _node_status_local()
        else:
            _c, res = call_node(n, ["--status", "--local"], timeout=120)
            per[n] = res
    local = per.get(LOCAL_NODE) or next(iter(per.values()))
    out = {"ok": True, "nodes": per, "node_order": nodes, "local_node": LOCAL_NODE,
           "suggested_path": local.get("suggested_path", ""),
           "archive_path": local.get("archive_path", ""),
           "configured": bool(local.get("configured")),
           "min_free_gb": local.get("min_free_gb", DEFAULT_MIN_FREE_GB),
           "cache_root": local.get("cache_root", CACHE_ROOT),
           "cache_free_bytes": local.get("cache_free_bytes", 0),
           "mounted": local.get("mounted", False),
           "archived": local.get("archived", [])}
    if "archive_free_bytes" in local:
        out["archive_free_bytes"] = local["archive_free_bytes"]
        out["archive_total_bytes"] = local.get("archive_total_bytes", 0)
    return 0, out


def op_list(scope="all"):
    try:
        nodes = node_names(scope)
    except ValueError as e:
        return 2, {"ok": False, "error": str(e)}
    per = {}
    for n in nodes:
        if n == LOCAL_NODE:
            rows = models_local()
            for r in rows:
                if r["archived"]:
                    r.update({k: v for k, v in fit_report(r["id"]).items() if k != "suggestions"})
            per[n] = {"ok": True, "models": rows, **_node_status_local()}
        else:
            _c, res = call_node(n, ["--list", "--local"], timeout=300)
            per[n] = res
    rows = merge_nodes(per)
    st = op_status(scope)[1]
    out = {"ok": True, "nodes": per, "node_order": nodes, "models": rows,
           "local_node": st.get("local_node", LOCAL_NODE),
           "suggested_path": st.get("suggested_path", ""),
           "archive_path": st.get("archive_path", ""),
           "configured": st.get("configured", False),
           "cache_free_bytes": st.get("cache_free_bytes", 0)}
    if "archive_free_bytes" in st:
        out["archive_free_bytes"] = st["archive_free_bytes"]
        out["archive_total_bytes"] = st.get("archive_total_bytes", 0)
    return 0, out


# ---------------------------------------------------------------------- output

def _print_nodes_plan(res):
    print(f"would archive {res['model_id']} ({res.get('engine', '?')}):")
    for n, p in res["nodes"].items():
        if not p.get("ok"):
            print(f"  {n:<7} REFUSED: {p.get('error')}")
            continue
        print(f"  {n:<7} {human(p.get('bytes')):>9} in {p.get('files', 0):>4} files, "
              f"frees ~{human(p.get('freed_est_bytes')):<9} -> {p.get('archive_path')} "
              f"(archive free {human(p.get('archive_free_bytes'))})")
    print(f"  total   {human(res.get('total_bytes')):>9} to copy, "
          f"~{human(res.get('total_freed_est_bytes'))} freed across nodes")


def _print_nodes_archive(res):
    print(f"archived {res['model_id']} ({res.get('engine', '?')}):")
    for n, p in res["nodes"].items():
        if not p.get("ok"):
            print(f"  {n:<7} FAILED: {p.get('error')}")
            continue
        print(f"  {n:<7} {human(p.get('bytes')):>9} copied, freed "
              f"{human(p.get('freed_bytes')):<9} (cache free "
              f"{human(p.get('free_after_bytes'))}) -> {p.get('archive_dir')}")
    print(f"  total   {human(res.get('total_bytes')):>9} copied, "
          f"freed {human(res.get('total_freed_bytes'))}")


def _print_nodes_unarchive(res):
    print(f"unarchived {res['model_id']}:")
    for n, p in res["nodes"].items():
        if not p.get("ok"):
            print(f"  {n:<7} FAILED: {p.get('error')}")
            continue
        print(f"  {n:<7} restored {human(p.get('restored_bytes')):<9} (cache free "
              f"{human(p.get('free_after_bytes'))})"
              + ("  archive copy pruned" if p.get("pruned") else ""))
    print(f"  total   {human(res.get('total_restored_bytes'))} restored")


def _print_unarchive_refusal(res):
    print(f"error: {res.get('error')}", file=sys.stderr)
    for n, p in res.get("nodes", {}).items():
        if not p.get("ok"):
            print(f"  {n:<7} {p.get('error')}", file=sys.stderr)
            continue
        mark = "ok" if p.get("fits") else f"SHORT by {human(p.get('shortfall_bytes'))}"
        print(f"  {n:<7} needs {human(p.get('needed_bytes')):<9} usable "
              f"{human(p.get('fits_have_bytes')):<9} {mark}", file=sys.stderr)
    printed = False
    for n, p in res.get("nodes", {}).items():
        for s in (p.get("suggestions") or []):
            if not printed:
                print("archive another model first, e.g.:", file=sys.stderr)
                printed = True
            print(f"  {n:<7} {s['id']}  (frees {human(s['frees_bytes'])})", file=sys.stderr)


def print_human(code, res):
    if not res.get("ok"):
        print(f"error: {res.get('error', 'failed')}", file=sys.stderr)
        if res.get("nodes"):
            if res.get("no_space_nodes"):
                _print_unarchive_refusal(res)
            else:
                for n, p in res["nodes"].items():
                    if not p.get("ok"):
                        print(f"  {n:<7} {p.get('error')}", file=sys.stderr)
        if res.get("rolled_back"):
            print(f"  rolled back staged copies on: {', '.join(res['rolled_back'])}",
                  file=sys.stderr)
        if res.get("hint"):
            print(f"  hint: {res['hint']}", file=sys.stderr)
        return code
    nodes_map = res.get("nodes") or {}
    if res.get("dry_run") and nodes_map and any("bytes" in p for p in nodes_map.values()
                                                if isinstance(p, dict)):
        _print_nodes_plan(res)
        return code
    if "models" in res:
        order = res.get("node_order") or list((res.get("nodes") or {}).keys())
        print(f"{'MODEL':48} {'ENGINE':6} {'STATE':9} "
              + " ".join(f"{n.upper():>9}" for n in order) + "  status")
        for m in res["models"]:
            per_node = m.get("nodes") or {}
            sizes = " ".join(f"{human((per_node.get(n) or {}).get('bytes') or 0):>9}"
                             for n in order)
            flags = []
            if m.get("current"):
                flags.append("current")
            if m.get("serving"):
                flags.append("live")
            if m["state"] == "partial":
                flags.append("HALF-ARCHIVED")
            print(f"{m['id']:<48} {m['engine']:<6} {m['state']:<9} {sizes}  {' '.join(flags)}")
        for n in order:
            info = (res.get("nodes") or {}).get(n) or {}
            if not info.get("ok"):
                print(f"\n{n}: {info.get('error', 'unavailable')}")
                continue
            ap = info.get("archive_path") or "(not configured)"
            if not info.get("archive_path") and info.get("suggested_path"):
                ap += f" - {info['suggested_path']} is mounted here; --set-path enables it"
            print(f"\n{n}: archive {ap}"
                  + (f"  free {human(info.get('archive_free_bytes'))}" if info.get("mounted") else "")
                  + f"  |  cache free {human(info.get('cache_free_bytes'))}")
        return code
    if res.get("nodes") and "total_freed_bytes" in res:
        _print_nodes_archive(res)
        return code
    if res.get("nodes") and "total_restored_bytes" in res:
        _print_nodes_unarchive(res)
        return code
    if "archive_path" in res and "models" not in res:
        print(f"archive path: {res['archive_path'] or '(not configured)'}")
        for n, info in nodes_map.items():
            if not isinstance(info, dict) or not info.get("ok"):
                print(f"  {n:<7} {info.get('error', 'unavailable')}")
                continue
            ap = info.get("archive_path") or "(not configured)"
            if not info.get("archive_path") and info.get("suggested_path"):
                ap += f"  ({info['suggested_path']} mounted here; --set-path enables it)"
            line = f"  {n:<7} {ap}"
            if info.get("mounted"):
                line += f"  free {human(info.get('archive_free_bytes'))}"
            line += f"  cache free {human(info.get('cache_free_bytes'))}"
            print(line)
        return code
    if res.get("nodes") and "failed_nodes" in res:
        print(f"error: {res.get('error')}", file=sys.stderr)
        for n, info in res["nodes"].items():
            print(f"  {n:<7} {(info.get('error') if not info.get('ok') else 'ok')}",
                  file=sys.stderr)
        return code
    if res.get("nodes") and res.get("path"):
        print(f"archive path set to {res['path']}:")
        for n, info in res["nodes"].items():
            free = human(info.get("free_bytes")) if info.get("free_bytes") else "?"
            print(f"  {n:<7} {free} free")
        return code
    if "path" in res:
        print(f"archive path set: {res['path']} ({human(res['free_bytes'])} free)")
        if res.get("suggested_path"):
            print(f"  (detected mount: {res['suggested_path']})")
        return code
    print(json.dumps(res, indent=2))
    return code


# ---------------------------------------------------------------------- plumbing

def build_parser():
    p = argparse.ArgumentParser(prog="vllm-model-archive",
                                description="archive/restore HF model caches across the DGX sparks")
    p.add_argument("--status", action="store_true", help="show archive config, per node")
    p.add_argument("--list", action="store_true", help="list models with per-node state")
    p.add_argument("--set-path", metavar="DIR", help="configure this node's archive location")
    p.add_argument("--archive", metavar="MODEL_ID", help="archive a model on every node")
    p.add_argument("--unarchive", metavar="MODEL_ID", help="restore a model on every node")
    p.add_argument("--prune-archive", action="store_true",
                   help="with --unarchive: delete the archive copy after a verified restore")
    p.add_argument("--node", default=None, choices=["all", "head", "worker"],
                   help="limit the operation to one node (default: all for "
                        "list/status/archive/unarchive, this node for --set-path)")
    p.add_argument("--local", action="store_true",
                   help="act on this node only (internal: used across nodes)")
    p.add_argument("--stage", default="commit",
                   choices=["plan", "copy", "commit", "rollback"], help=argparse.SUPPRESS)
    p.add_argument("--dry-run", action="store_true", help="report what would happen")
    p.add_argument("--self-node", metavar="NAME", help=argparse.SUPPRESS)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    return p


def dispatch(argv):
    """Parse argv and run one operation. Returns (exit-code, result-dict)."""
    a = build_parser().parse_args(list(argv))
    if a.self_node:
        set_local_node(a.self_node)
    scope = LOCAL_NODE if a.local else (a.node or "all")
    try:
        if a.set_path:
            if a.local or a.node is None:
                code, res = op_set_path(a.set_path)
            else:
                code, res = op_set_path_cluster(a.set_path, a.node)
        elif a.archive:
            if a.local:
                code, res = archive_local(a.archive, stage=a.stage, dry_run=a.dry_run)
            else:
                code, res = op_archive(a.archive, dry_run=a.dry_run, scope=scope)
        elif a.unarchive:
            if a.local:
                code, res = unarchive_local(a.unarchive, stage=a.stage, dry_run=a.dry_run,
                                            prune=a.prune_archive)
            else:
                code, res = op_unarchive(a.unarchive, dry_run=a.dry_run,
                                         prune=a.prune_archive, scope=scope)
        elif a.list:
            code, res = op_list(scope)
        else:
            code, res = op_status(scope)
    except OSError as e:
        code, res = 4, {"ok": False, "error": f"I/O error: {e}"}
    return code, res


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    want_json = "--json" in argv
    code, res = dispatch(argv)
    if want_json:
        print(json.dumps(res, indent=2, sort_keys=True))
    else:
        code = print_human(code, res)
    return code


if __name__ == "__main__":
    sys.exit(main())
