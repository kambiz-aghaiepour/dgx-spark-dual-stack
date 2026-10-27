#!/usr/bin/env python3
"""vllm-model-archive — move HF checkpoint caches between internal storage and an
archive disk, keeping the model profiles (flags/config) in place.

The internal cache is small (one or two checkpoints).  The archive location holds
one directory per model/engine with a self-contained copy of that model's cache
tree plus a manifest, so it can be pulled back later.

Layout:

    $ARCHIVE_PATH/models/<safe-name>/
        manifest.json          what/when/how big, per node
        hub/                   cache-root-relative tree that was archived
            models--org--name/...
            blobs/xx/<sha>     external blob targets, only when referenced

Nothing outside the HF cache is touched: flags profiles
(`~/.config/spark-vllm/flags/*`), `model.conf` and `locked.conf` stay as they are,
which is what makes an archived model still listed, lockable and re-selectable.

CLI (add --json for machine-readable output):

    vllm-model-archive --status                     # show archive config
    vllm-model-archive --set-path /archive          # configure the archive path
    vllm-model-archive --list                        # known models + state
    vllm-model-archive --archive <model-id> [--dry-run]
    vllm-model-archive --unarchive <model-id> [--dry-run] [--prune-archive]

Exit codes: 0 ok · 2 refused/precondition · 3 does not fit · 4 I/O failure
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

TOOL = "vllm-model-archive 0.1.0"
CACHE_ROOT = os.path.expanduser("~/.cache/huggingface/hub")
CONF_DIR = os.path.expanduser("~/.config/spark-vllm")
ARCHIVE_CONF = os.path.join(CONF_DIR, "archive.conf")
MODEL_CONF = os.path.join(CONF_DIR, "model.conf")
FLAGS_DIR = os.path.join(CONF_DIR, "flags")
DEFAULT_MIN_FREE_GB = 10.0


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


def models(path=None):
    """Every model the tool knows: cached, archived, or both."""
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
                        "cache_bytes": payload(mid)["bytes"]}
    for mid, m in arch.items():
        row = out.setdefault(mid, {"id": mid, "engine": m.get("engine", engine_for(mid)),
                                   "cached": False, "archived": True,
                                   "profiles": profile_files(mid), "cache_bytes": 0})
        row["archived"] = True
        row["archive_bytes"] = m.get("nodes", {}).get("head", {}).get("payload_bytes", 0)
        row["archived_at"] = m.get("created_at", "")
    for row in out.values():
        row["state"] = ("cached+archived" if row["cached"] and row["archived"]
                        else "cached" if row["cached"] else "archived")
        row["current"] = row["id"] == boot_model()
    return [out[k] for k in sorted(out)]


def fit_report(mid, path=None):
    """Can <mid> be pulled back into the internal cache right now?"""
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
        cands = [m for m in models(path)
                 if m["cached"] and not m["current"] and m["id"] != mid]
        cands.sort(key=lambda m: m["cache_bytes"], reverse=True)
        acc, picked = 0, []
        for m in cands:
            picked.append(m)
            acc += m["cache_bytes"]
            if acc >= shortfall:
                break
        # Best effort even when no combination is enough: the user still needs
        # to see what the biggest wins would be.
        sufficient = acc >= shortfall
        suggestions = [{"id": m["id"], "frees_bytes": m["cache_bytes"]}
                       for m in picked]
    return {"fits": fits, "needed_bytes": needed, "free_bytes": free,
            "margin_bytes": margin, "fits_have_bytes": free - margin,
            "shortfall_bytes": max(0, needed + margin - free),
            "suggestions_sufficient": sufficient,
            "suggestions": suggestions,
            "archive_bytes": man.get("nodes", {}).get("head", {}).get("payload_bytes", 0),
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
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=3) as r:
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


# ----------------------------------------------------------------- operations

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


def op_archive(mid, dry_run=False):
    conf = archive_conf()
    if not conf["path"]:
        return 2, {"ok": False, "error": "archive path not configured (run --set-path /archive)"}
    if not os.path.isdir(conf["path"]):
        return 2, {"ok": False, "error": f"archive path {conf['path']} is not mounted"}
    rd = repo_dir(mid)
    if not os.path.isdir(rd):
        if os.path.exists(os.path.join(model_archive_dir(mid), "manifest.json")):
            return 2, {"ok": False, "error": f"{mid} is already archived"}
        return 2, {"ok": False, "error": f"{mid} is not in the cache ({rd})"}
    if mid == boot_model():
        return 2, {"ok": False,
                   "error": f"{mid} is the boot default (model.conf) - select another model first"}
    live = served_model()
    if live and live == mid and service_active() == "active":
        return 2, {"ok": False,
                   "error": f"{mid} is being served right now - stop or switch the service first"}
    pay = payload(mid)
    if not pay["entries"]:
        return 2, {"ok": False, "error": f"nothing to archive under {rd}"}
    adir = model_archive_dir(mid)
    hub = os.path.join(adir, "hub")
    disk = shutil.disk_usage(conf["path"])
    if disk.free < pay["bytes"]:
        return 3, {"ok": False, "error": "archive disk full",
                   "needed_bytes": pay["bytes"], "free_bytes": disk.free}
    if dry_run:
        return 0, {"ok": True, "dry_run": True, "model_id": mid,
                   "engine": engine_for(mid), "bytes": pay["bytes"],
                   "files": pay["files"], "externals": len(pay["externals"]),
                   "dangling": pay["dangling"],
                   "archive_dir": adir, "free_on_archive": disk.free}

    os.makedirs(hub, exist_ok=True)
    manifest = {
        "schema": 1, "tool": TOOL, "model_id": mid, "engine": engine_for(mid),
        "state": "archiving", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename, "archive_path": conf["path"],
        "profiles": [os.path.basename(p) for p in profile_files(mid)],
        "nodes": {"head": {"cache_root": CACHE_ROOT, "repo_rel": repo_rel(mid),
                           "payload_bytes": pay["bytes"], "payload_files": pay["files"],
                           "externals": pay["externals"]}},
    }
    with open(os.path.join(adir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    rels = pay["entries"] + pay["externals"]
    method, _ncopied, cerr = copy_paths(CACHE_ROOT, hub, rels)
    if cerr:
        return 4, {"ok": False, "error": f"{method} copy failed: {cerr}"}
    bad = verify_tree(CACHE_ROOT, hub, rels)
    if bad:
        return 4, {"ok": False, "error": "copy verification failed; source left intact",
                   "detail": ", ".join(bad[:5])}

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
    manifest["state"] = "archived"
    manifest["nodes"]["head"]["freed_bytes"] = freed
    manifest["nodes"]["head"]["df_delta_bytes"] = max(0, after - before)
    with open(os.path.join(adir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return 0, {"ok": True, "model_id": mid, "engine": engine_for(mid),
               "bytes": pay["bytes"], "files": pay["files"], "copy_method": method,
               "externals_removed": removed_ext, "freed_bytes": freed,
               "df_delta_bytes": max(0, after - before),
               "free_after_bytes": shutil.disk_usage(CACHE_ROOT).free,
               "archive_dir": adir}


def op_unarchive(mid, dry_run=False, prune_archive=False):
    conf = archive_conf()
    adir = model_archive_dir(mid)
    mf = os.path.join(adir, "manifest.json")
    if not os.path.exists(mf):
        return 2, {"ok": False, "error": f"{mid} is not archived under {conf['path'] or '(unset)'}"}
    fit = fit_report(mid)
    if not fit["fits"] and not dry_run:
        return 3, {"ok": False, "error": "not enough internal space", **fit}
    hub = os.path.join(adir, "hub")
    rels = []
    for root, _dirs, names in os.walk(hub):
        for n in names:
            rels.append(os.path.relpath(os.path.join(root, n), hub))
    if dry_run:
        return 0, {"ok": True, "dry_run": True, "model_id": mid, **fit,
                   "files": len(rels)}
    os.makedirs(CACHE_ROOT, exist_ok=True)
    method, _ncopied, cerr = copy_paths(hub, CACHE_ROOT, rels)
    if cerr:
        return 4, {"ok": False, "error": f"{method} restore failed: {cerr}"}
    bad = verify_tree(hub, CACHE_ROOT, rels)
    if bad:
        return 4, {"ok": False, "error": "restore verification failed",
                   "detail": ", ".join(bad[:5])}
    with open(mf) as f:
        man = json.load(f)
    man["state"] = "restored"
    man["restored_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(mf, "w") as f:
        json.dump(man, f, indent=2, sort_keys=True)
    if prune_archive:
        shutil.rmtree(adir, ignore_errors=True)
    return 0, {"ok": True, "model_id": mid, "restored_bytes": fit["needed_bytes"],
               "files": len(rels), "pruned": bool(prune_archive),
               "free_after_bytes": shutil.disk_usage(CACHE_ROOT).free}


def op_status():
    conf = archive_conf()
    out = {"archive_path": conf["path"], "configured": bool(conf["path"]),
           "min_free_gb": conf["min_free_gb"],
           "cache_root": CACHE_ROOT,
           "cache_free_bytes": shutil.disk_usage(CACHE_ROOT).free}
    if conf["path"] and os.path.isdir(conf["path"]):
        d = shutil.disk_usage(conf["path"])
        out.update({"mounted": True, "archive_free_bytes": d.free, "archive_total_bytes": d.total})
        out["archived"] = sorted(scan_archived(conf["path"]))
    else:
        out.update({"mounted": False, "archived": []})
    return 0, {"ok": True, **out}


def op_list():
    conf = archive_conf()
    rows = models()
    for r in rows:
        if r["archived"]:
            r.update({k: v for k, v in fit_report(r["id"]).items() if k != "suggestions"})
    out = {"ok": True, "archive_path": conf["path"], "configured": bool(conf["path"]),
           "cache_free_bytes": shutil.disk_usage(CACHE_ROOT).free, "models": rows}
    if conf["path"] and os.path.isdir(conf["path"]):
        d = shutil.disk_usage(conf["path"])
        out.update({"archive_free_bytes": d.free, "archive_total_bytes": d.total})
    return 0, out


# ---------------------------------------------------------------------- plumbing

def print_human(code, res):
    if not res.get("ok"):
        print(f"error: {res.get('error', 'failed')}", file=sys.stderr)
        if res.get("suggestions"):
            print("archive another model first, e.g.:", file=sys.stderr)
            for s in res["suggestions"]:
                print(f"  {s['id']}  (frees {human(s['frees_bytes'])})", file=sys.stderr)
        if res.get("needed_bytes") and res.get("free_bytes") is not None:
            print(f"  needed {human(res['needed_bytes'])}, usable free "
                  f"{human(res.get('fits_have_bytes', res['free_bytes']))}", file=sys.stderr)
        return code
    if "models" in res:
        print(f"{'MODEL':60} {'ENGINE':6} {'STATE':16} {'CACHED':>9} {'ARCHIVE':>9}")
        for m in res["models"]:
            print(f"{m['id']:<60} {m['engine']:<6} {m['state']:<16} "
                  f"{human(m['cache_bytes']):>9} {human(m.get('archive_bytes', 0)):>9}")
        print(f"\narchive: {res['archive_path'] or '(not configured)'}")
        return code
    if "archive_path" in res and "models" not in res:
        print(f"archive path: {res['archive_path'] or '(not configured)'}")
        if res.get("mounted"):
            print(f"archive free: {human(res['archive_free_bytes'])} of {human(res['archive_total_bytes'])}")
            print("archived:     " + (", ".join(res["archived"]) or "(none)"))
        print(f"cache free:   {human(res['cache_free_bytes'])}")
        return code
    if res.get("dry_run"):
        print(f"would archive {res['model_id']} ({res['engine']}): "
              f"{human(res['bytes'])} in {res['files']} files -> {res['archive_dir']}")
        return code
    if "freed_bytes" in res:
        print(f"archived {res['model_id']} ({res['engine']}): {human(res['bytes'])}, "
              f"freed {human(res['freed_bytes'])} internal, {res['files']} files")
        print(f"  archive: {res['archive_dir']}")
        return code
    if "restored_bytes" in res:
        print(f"unarchived {res['model_id']}: {human(res['restored_bytes'])} restored"
              + (" (archive copy pruned)" if res.get("pruned") else ""))
        return code
    if "path" in res:
        print(f"archive path set: {res['path']} ({human(res['free_bytes'])} free)")
        return code
    print(json.dumps(res, indent=2))
    return code


def main(argv=None):
    p = argparse.ArgumentParser(prog="vllm-model-archive",
                                description="archive/restore HF model caches on the DGX sparks")
    p.add_argument("--status", action="store_true", help="show archive config and usage")
    p.add_argument("--list", action="store_true", help="list known models and their state")
    p.add_argument("--set-path", metavar="DIR", help="configure the archive location")
    p.add_argument("--archive", metavar="MODEL_ID", help="archive a model's cached data")
    p.add_argument("--unarchive", metavar="MODEL_ID", help="restore a model's cached data")
    p.add_argument("--prune-archive", action="store_true",
                   help="with --unarchive: delete the archive copy after a verified restore")
    p.add_argument("--dry-run", action="store_true", help="report what would happen")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    a = p.parse_args(argv)

    try:
        if a.set_path:
            code, res = op_set_path(a.set_path)
        elif a.archive:
            code, res = op_archive(a.archive, dry_run=a.dry_run)
        elif a.unarchive:
            code, res = op_unarchive(a.unarchive, dry_run=a.dry_run, prune_archive=a.prune_archive)
        elif a.list:
            code, res = op_list()
        else:
            code, res = op_status()
    except OSError as e:
        code, res = 4, {"ok": False, "error": f"I/O error: {e}"}

    if a.json:
        print(json.dumps(res, indent=2, sort_keys=True))
    else:
        code = print_human(code, res)
    return code


if __name__ == "__main__":
    sys.exit(main())
