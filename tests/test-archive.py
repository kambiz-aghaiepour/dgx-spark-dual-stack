#!/usr/bin/env python3
"""Synthetic tests for scripts/vllm-model-archive.py — no real models, no archive disk.

Run:  python3 tests/test-archive.py
The archive location defaults to /dev/shm (needs a different filesystem than
$HOME); override with ARCHIVE_TEST_PATH=/some/other/fs/dir.
"""
import hashlib
import http.server
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(REPO, "scripts", "vllm-model-archive.py")
BASE = tempfile.mkdtemp(prefix="vllm-archtest-")
HOME = os.path.join(BASE, "home")
HUB = os.path.join(HOME, ".cache/huggingface/hub")
SHM = os.environ.get("ARCHIVE_TEST_PATH") or "/dev/shm/vllm-archtest"
ARCH = os.path.join(SHM, "archive")

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def run(*args, expect=0, home=HOME):
    env = dict(os.environ, HOME=home)
    r = subprocess.run([sys.executable, HELPER, *args], capture_output=True, text=True, env=env)
    if expect is not None and r.returncode != expect:
        print(f"   !! {args} -> rc={r.returncode}\n   stdout={r.stdout[:400]}\n   stderr={r.stderr[:400]}")
    return r


def jrun(*args, **kw):
    r = run(*args, "--json", **kw)
    try:
        return r.returncode, json.loads(r.stdout)
    except ValueError:
        return r.returncode, {"raw": r.stdout, "err": r.stderr}


def write(p, data=b"", size=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(p, mode) as f:
        f.write(data)
    if size:
        with open(p, "wb") as f:
            f.truncate(size)


def build():
    shutil.rmtree(BASE, ignore_errors=True)
    shutil.rmtree(SHM, ignore_errors=True)
    os.makedirs(HUB)
    os.makedirs(os.path.join(HOME, ".config/spark-vllm/flags"))
    os.makedirs(ARCH)
    os.makedirs(os.path.join(BASE, "arch-samefs"))
    # --- model alpha: all bytes inside the repo dir
    a = os.path.join(HUB, "models--org--alpha")
    write(os.path.join(a, "refs/main"), "abc123\n")
    write(os.path.join(a, "blobs/aaa1"), b"A" * 4096)
    write(os.path.join(a, "blobs/aaa2"), b"B" * 2048)
    os.makedirs(os.path.join(a, "snapshots/abc123"))
    for n, t in (("weights.bin", "aaa1"), ("config.json", "aaa2")):
        os.symlink(f"../../blobs/{t}", os.path.join(a, "snapshots/abc123", n))
    # --- model beta: weights live in the shared (Xet-style) blob store
    shared = os.path.join(HUB, "blobs/aa", "deadbeef" * 5)
    write(shared, b"", size=1_000_000)
    b = os.path.join(HUB, "models--org--beta")
    write(os.path.join(b, "refs/main"), "def456\n")
    os.makedirs(os.path.join(b, "blobs"))
    os.symlink(os.path.relpath(shared, os.path.join(b, "blobs")), os.path.join(b, "blobs/cee1"))
    write(os.path.join(b, "blobs/cee2"), b"cfg")
    os.makedirs(os.path.join(b, "snapshots/def456"))
    for n, t in (("weights.bin", "cee1"), ("config.json", "cee2")):
        os.symlink(f"../../blobs/{t}", os.path.join(b, "snapshots/def456", n))
    # --- model gamma: shares beta's external blob (must survive beta's archive)
    g = os.path.join(HUB, "models--org--gamma")
    write(os.path.join(g, "refs/main"), "999\n")
    os.makedirs(os.path.join(g, "blobs"))
    os.symlink(os.path.relpath(shared, os.path.join(g, "blobs")), os.path.join(g, "blobs/cee1"))
    os.makedirs(os.path.join(g, "snapshots/999"))
    os.symlink("../../blobs/cee1", os.path.join(g, "snapshots/999", "weights.bin"))
    # --- model delta: independent big-ish model, archive candidate for suggestions
    d = os.path.join(HUB, "models--org--delta")
    write(os.path.join(d, "refs/main"), "777\n")
    write(os.path.join(d, "blobs/ddd1"), b"", size=3_000_000)
    os.makedirs(os.path.join(d, "snapshots/777"))
    os.symlink("../../blobs/ddd1", os.path.join(d, "snapshots/777", "weights.bin"))
    # profiles (must survive archiving)
    for mid in ("org-alpha", "org-beta", "org-delta"):
        write(os.path.join(HOME, ".config/spark-vllm/flags", mid + ".conf"), "-tp 2\n")
    write(os.path.join(HOME, ".config/spark-vllm/model.conf"), "MODEL_ID=org/delta\n")
    return shared


def tree_state(root):
    out = {}
    for r, _d, names in os.walk(root):
        for n in names:
            p = os.path.join(r, n)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out[rel] = "link:" + os.readlink(p)
            else:
                with open(p, "rb") as f:
                    out[rel] = hashlib.sha256(f.read()).hexdigest()[:12] + f":{os.path.getsize(p)}"
    return out


def main():
    shared = build()
    beta_snap = os.path.join(HUB, "models--org--beta/snapshots/def456/weights.bin")
    alpha_snap = os.path.join(HUB, "models--org--alpha/snapshots/abc123/config.json")
    check("fixture: beta snapshot readable", open(beta_snap, "rb").read()[:1] == b"\0" or os.path.getsize(beta_snap) == 1_000_000)

    rc, res = jrun("--status")
    check("status not configured", rc == 0 and res["configured"] is False, res)
    rc, res = jrun("--archive", "org/alpha", expect=2)
    check("archive refused without path", rc == 2 and "not configured" in res.get("error", ""), res)
    # the current-model rule wins over the missing-config rule: configuring a
    # path would not make archiving the active model legal
    rc, res = jrun("--archive", "org/delta", expect=2)
    check("current model refused even with no archive path",
          rc == 2 and "cannot archive the current model" in res.get("error", ""), res)

    # path validation
    rc, res = jrun("--set-path", os.path.join(BASE, "arch-samefs"), expect=2)
    check("same-fs path refused", rc == 2 and "same filesystem" in res.get("error", ""), res)
    rc, res = jrun("--set-path", "/nonexistent/archive", expect=2)
    check("missing path refused", rc == 2, res)
    rc, res = jrun("--set-path", ARCH)
    check("set-path ok", rc == 0 and res["path"] == ARCH, res)

    rc, res = jrun("--list")
    ids = {m["id"]: m for m in res["models"]}
    check("list has 4 models", len(ids) == 4, list(ids))
    check("engine mapping miaai/b12x",
          all(m["engine"] == "b12x" for m in ids.values()))
    check("current flag on delta", ids["org/delta"]["current"] is True)
    check("active flag only on delta",
          ids["org/delta"]["active"] is True
          and not any(m["active"] for k, m in ids.items() if k != "org/delta"))

    rc, res = jrun("--archive", "org/delta", expect=2)
    err = res.get("error", "")
    check("refused: current model is the boot default",
          rc == 2 and "cannot archive the current model" in err
          and "switch to another model/engine first" in err and res.get("reason") == "boot default",
          res)

    # while the engine serves a model, archiving that model is refused but any
    # other model may still be archived
    class _Eng(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": "org/alpha"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _Eng)
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    os.environ["SPARK_VLLM_ENGINE_URL"] = f"http://127.0.0.1:{srv.server_address[1]}"
    rc, res = jrun("--list")
    live = {m["id"]: m for m in res.get("models", [])}
    check("serving flag from the engine", live.get("org/alpha", {}).get("serving") is True
          and live.get("org/alpha", {}).get("active") is True, live.get("org/alpha"))
    rc, res = jrun("--archive", "org/alpha", expect=2)
    err = res.get("error", "")
    check("refused: current model is being served",
          rc == 2 and "cannot archive the current model" in err
          and "serving it right now" in err and res.get("reason") == "serving", res)
    rc, res = jrun("--archive", "org/beta", "--dry-run")
    check("other models stay archivable while one is served", rc == 0, res)
    srv.shutdown()
    os.environ.pop("SPARK_VLLM_ENGINE_URL", None)

    # --- archive alpha (repo-internal payload)
    alpha_tree = tree_state(os.path.join(HUB, "models--org--alpha"))
    rc, res = jrun("--archive", "org/alpha", "--dry-run")
    check("dry-run alpha", rc == 0 and res["bytes"] > 6000 and res["files"] == 3, res)
    rc, res = jrun("--archive", "org/alpha")
    check("archive alpha ok", rc == 0 and res["freed_bytes"] > 0, res)
    check("alpha repo removed", not os.path.exists(os.path.join(HUB, "models--org--alpha")))
    adir = os.path.join(ARCH, "models", "org-alpha")
    check("alpha manifest exists", os.path.exists(os.path.join(adir, "manifest.json")))
    pre = "models--org--alpha/"
    arch_tree = {k[len(pre):] if k.startswith(pre) else k: v
                 for k, v in tree_state(os.path.join(adir, "hub")).items()}
    diff = {k: (alpha_tree.get(k), arch_tree.get(k)) for k in set(alpha_tree) | set(arch_tree)
            if alpha_tree.get(k) != arch_tree.get(k)}
    check("alpha archive content identical", not diff, diff)
    check("alpha profile kept", os.path.exists(os.path.join(HOME, ".config/spark-vllm/flags/org-alpha.conf")))
    man = json.load(open(os.path.join(adir, "manifest.json")))
    check("manifest engine/size", man["engine"] == "b12x" and man["state"] == "archived"
          and man["nodes"]["head"]["payload_bytes"] == res["bytes"], man.get("state"))

    rc, res = jrun("--list")
    ids = {m["id"]: m for m in res["models"]}
    check("alpha listed as archived", ids["org/alpha"]["state"] == "archived"
          and ids["org/alpha"]["cached"] is False, ids["org/alpha"])
    check("alpha fit ok now", ids["org/alpha"]["fits"] is True, ids["org/alpha"])
    rc, res = jrun("--archive", "org/alpha", expect=2)
    check("re-archive refused", rc == 2 and "already archived" in res.get("error", ""), res)

    # --- unarchive alpha, compare bytes
    rc, res = jrun("--unarchive", "org/alpha")
    check("unarchive alpha ok", rc == 0, res)
    check("alpha restored identical", tree_state(os.path.join(HUB, "models--org--alpha")) == alpha_tree)
    check("alpha snapshot resolves", os.path.getsize(alpha_snap) == 2048)

    # --- archive beta: external blob shared with gamma must survive
    rc, res = jrun("--archive", "org/beta")
    check("archive beta ok", rc == 0, res)
    check("shared external blob kept", os.path.exists(shared))
    check("gamma repo intact", os.path.exists(os.path.join(HUB, "models--org--gamma/snapshots/999/weights.bin")))
    gsnap = os.path.join(HUB, "models--org--gamma/snapshots/999/weights.bin")
    check("gamma weights still readable", os.path.getsize(gsnap) == 1_000_000, os.path.getsize(gsnap) if os.path.exists(gsnap) else "missing")
    bman = json.load(open(os.path.join(ARCH, "models/org-beta/manifest.json")))
    check("beta manifest records external", bman["nodes"]["head"]["externals"] != [], bman["nodes"]["head"])
    rc, res = jrun("--unarchive", "org/beta")
    check("unarchive beta ok (shared blob reused)", rc == 0, res)
    check("beta weights restored", os.path.getsize(beta_snap) == 1_000_000)

    # --- no-space path: huge margin forces refusal with suggestions
    conf = os.path.join(HOME, ".config/spark-vllm/archive.conf")
    txt = open(conf).read().replace("MIN_FREE_GB=10.0", "MIN_FREE_GB=100000000.0")
    open(conf, "w").write(txt)
    rc, res = jrun("--archive", "org/delta")   # delta is boot model -> refused, pick alpha instead
    jrun("--archive", "org/alpha")
    rc, res = jrun("--unarchive", "org/alpha", expect=3)
    check("unarchive refused when no space", rc == 3 and res.get("fits") is False, res)
    check("suggestions offered", bool(res.get("suggestions")), res.get("suggestions"))
    txt = open(conf).read().replace("MIN_FREE_GB=100000000.0", "MIN_FREE_GB=10.0")
    open(conf, "w").write(txt)

    # --- prune archive
    rc, res = jrun("--unarchive", "org/alpha", "--prune-archive")
    check("prune after restore", rc == 0 and not os.path.exists(os.path.join(ARCH, "models/org-alpha")), res)

    # --- human output smoke
    r = run("--status")
    check("human status prints", r.returncode == 0 and "archive path" in r.stdout, r.stdout)
    r = run("--list")
    check("human list prints", r.returncode == 0 and "org/gamma" in r.stdout, r.stdout[:200])

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
