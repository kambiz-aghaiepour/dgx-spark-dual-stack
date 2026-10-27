#!/usr/bin/env python3
"""Synthetic cluster tests for scripts/vllm-model-archive.py.

No real models, no archive disks, no ssh: two throwaway HOMEs stand in for the
two sparks and tests/fake-node.sh is wired in through SPARK_VLLM_REMOTE_CMD, so
the orchestrator still crosses a real subprocess boundary for the worker.

Run:  python3 tests/test-archive.py
Archive dirs default to /dev/shm (must be a different filesystem than $TMPDIR);
override with ARCHIVE_TEST_PATH=/some/other/fs/dir.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(REPO, "scripts", "vllm-model-archive.py")
SHIM = os.path.join(REPO, "tests", "fake-node.sh")
BASE = tempfile.mkdtemp(prefix="vllm-archtest-")
HEAD_HOME = os.path.join(BASE, "head")
WORK_HOME = os.path.join(BASE, "worker")
SHM = os.environ.get("ARCHIVE_TEST_PATH") or "/dev/shm/vllm-archtest"
HEAD_ARCH = os.path.join(SHM, "head-archive")
WORK_ARCH = os.path.join(SHM, "worker-archive")

fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def env_for(home, cluster=True, extra=None):
    env = dict(os.environ, HOME=home, SPARK_VLLM_ENGINE_URL="http://127.0.0.1:9")
    if cluster:
        env.setdefault("SPARK_VLLM_REMOTE_CMD", SHIM + " {host} {helper} {args}")
        env.setdefault("SPARK_VLLM_ARCHIVE_HELPER", HELPER)
        env.setdefault("FAKE_NODE_HOME", WORK_HOME)
        env.setdefault("WORKER", "worker")
    if extra:
        env.update(extra)
    return env


def run(*args, home=HEAD_HOME, expect=0, cluster=True, extra=None):
    r = subprocess.run([sys.executable, HELPER, *args], capture_output=True, text=True,
                       env=env_for(home, cluster, extra))
    if expect is not None and r.returncode != expect:
        print(f"   !! {args} -> rc={r.returncode}\n   stdout={r.stdout[:700]}\n   stderr={r.stderr[:400]}")
    return r


def jrun(*args, **kw):
    r = run(*args, "--json", **kw)
    try:
        return r.returncode, json.loads(r.stdout)
    except ValueError:
        return r.returncode, {"raw": r.stdout, "err": r.stderr}


def write(path, data=b"", size=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data if isinstance(data, bytes) else data.encode())
    if size:
        with open(path, "wb") as f:
            f.truncate(size)


def node_conf(home, path, min_free_gb=10):
    write(os.path.join(home, ".config/spark-vllm/archive.conf"),
          f"ARCHIVE_PATH={path}\nMIN_FREE_GB={min_free_gb}\n")


def build(home, specs, boot="org/delta"):
    """specs: {model_id: (n_files, size_each)} — every file is repo-internal."""
    hub = os.path.join(home, ".cache/huggingface/hub")
    os.makedirs(os.path.join(home, ".config/spark-vllm/flags"), exist_ok=True)
    write(os.path.join(home, ".config/spark-vllm/model.conf"), f"MODEL_ID={boot}\n")
    for mid, (n, each) in specs.items():
        rd = os.path.join(hub, "models--" + mid.replace("/", "--"))
        snap = os.path.join(rd, "snapshots", "abc")
        os.makedirs(snap, exist_ok=True)
        write(os.path.join(rd, "refs", "main"), b"")
        for i in range(n):
            target = os.path.join(rd, "blobs", f"sha{i}")
            write(target, b"x", size=each)
            os.symlink(os.path.relpath(target, snap), os.path.join(snap, f"w{i}.safetensors"))
    return hub


def tree(path):
    out = {}
    for root, _d, names in os.walk(path):
        for n in names:
            p = os.path.join(root, n)
            rel = os.path.relpath(p, path)
            out[rel] = -1 if os.path.islink(p) else os.path.getsize(p)
    return out


def main():
    for d in (BASE, SHM):
        shutil.rmtree(d, ignore_errors=True)
    for d in (HEAD_ARCH, WORK_ARCH):
        os.makedirs(d, exist_ok=True)
    os.chmod(SHIM, 0o755)

    # deliberately different content per node: head has small copies, the
    # worker full ones — exactly the asymmetry the real pair shows
    head_hub = build(HEAD_HOME, {"org/alpha": (2, 8192), "org/beta": (1, 4096),
                                "org/delta": (1, 2048)})
    work_hub = build(WORK_HOME, {"org/alpha": (3, 16384), "org/beta": (1, 8192),
                                "org/delta": (1, 4096)})

    # --- an /archive mount is offered, but is not "configured" by itself
    rc, res = jrun("--status", home=WORK_HOME, cluster=False,
                   extra={"SPARK_VLLM_DEFAULT_ARCHIVE": "/dev/shm"})
    check("detected mount is suggested", res.get("suggested_path") == "/dev/shm",
          res.get("suggested_path"))
    check("detected mount does not configure", res.get("configured") is False, res)
    r = subprocess.run([sys.executable, HELPER, "--status"],
                       capture_output=True, text=True,
                       env=env_for(WORK_HOME, cluster=False,
                                   extra={"SPARK_VLLM_DEFAULT_ARCHIVE": "/dev/shm"}))
    check("status says the mount is available but off",
          "/dev/shm mounted here" in r.stdout, r.stdout.strip())

    # --- per-node configuration
    rc, res = jrun("--set-path", WORK_ARCH, "--node", "worker")
    check("set-path on the worker from the head",
          rc == 0 and res.get("nodes", {}).get("worker", {}).get("ok") is True, res)
    with open(os.path.join(WORK_HOME, ".config/spark-vllm/archive.conf")) as f:
        check("worker archive.conf written", f"ARCHIVE_PATH={WORK_ARCH}" in f.read())
    rc, res = jrun("--set-path", "/nonexistent/archive", "--node", "worker", expect=2)
    check("bad worker path refused, naming the node",
          rc == 2 and "worker" in res.get("error", ""), res)
    rc, res = jrun("--set-path", HEAD_ARCH, "--node", "all")
    check("set-path --node all configures both",
          rc == 0 and set(res.get("nodes", {})) == {"head", "worker"}, res)

    rc, res = jrun("--set-path", HEAD_ARCH, home=HEAD_HOME, cluster=False)
    check("head set-path", rc == 0 and res.get("path") == HEAD_ARCH, res)
    rc, res = jrun("--set-path", WORK_ARCH, home=WORK_HOME, cluster=False)
    check("worker set-path", rc == 0 and res.get("path") == WORK_ARCH, res)

    rc, res = jrun("--status")
    nodes = res.get("nodes", {})
    check("status lists both nodes", set(nodes) == {"head", "worker"}, list(nodes))
    check("status: per-node archive paths",
          nodes["head"].get("archive_path") == HEAD_ARCH
          and nodes["worker"].get("archive_path") == WORK_ARCH, nodes)
    check("status: free space per node",
          all("archive_free_bytes" in n and "cache_free_bytes" in n for n in nodes.values()), nodes)

    # --- listing merges both caches
    rc, res = jrun("--list")
    rows = {m["id"]: m for m in res["models"]}
    check("list: node order", res.get("node_order") == ["head", "worker"], res.get("node_order"))
    hb = rows["org/alpha"]["nodes"]["head"]["bytes"]
    wb = rows["org/alpha"]["nodes"]["worker"]["bytes"]
    check("list: sizes reported per node (head 2 files, worker 3 bigger ones)",
          hb >= 2 * 8192 and wb >= 3 * 16384 and wb > hb, (hb, wb))
    check("list: alpha cached on both", rows["org/alpha"]["cached"] is True)
    check("list: delta is current", rows["org/delta"]["current"] is True)

    # --- the active model is refused before anything moves
    rc, res = jrun("--archive", "org/delta", expect=2)
    check("refused: current model (cluster)",
          rc == 2 and "cannot archive the current model" in res.get("error", ""), res)
    check("refused: nothing staged", not os.path.exists(os.path.join(HEAD_ARCH, "models")),
          os.listdir(HEAD_ARCH))

    # --- dry run reports per node
    rc, res = jrun("--archive", "org/alpha", "--dry-run")
    n = res.get("nodes", {})
    check("dry-run: per-node bytes",
          n["head"]["bytes"] >= 16384 and n["worker"]["bytes"] >= 49152
          and n["worker"]["bytes"] > n["head"]["bytes"], n)
    check("dry-run: total is the sum of the nodes",
          res.get("total_bytes") == n["head"]["bytes"] + n["worker"]["bytes"],
          res.get("total_bytes"))

    # --- real archive: copy both, verify both, then delete both
    rc, res = jrun("--archive", "org/alpha")
    got = res.get("nodes", {})
    check("archive: both nodes ok", rc == 0 and all(v.get("ok") for v in got.values()), res)
    check("archive: per-node freed",
          got["head"]["freed_bytes"] >= 16384 and got["worker"]["freed_bytes"] >= 49152, got)
    check("archive: total freed is the sum",
          res.get("total_freed_bytes") == got["head"]["freed_bytes"] + got["worker"]["freed_bytes"],
          res.get("total_freed_bytes"))
    check("archive: head copy gone", not os.path.exists(os.path.join(head_hub, "models--org--alpha")))
    check("archive: worker copy gone", not os.path.exists(os.path.join(work_hub, "models--org--alpha")))
    check("archive: head tree present",
          os.path.exists(os.path.join(HEAD_ARCH, "models/org-alpha/hub/models--org--alpha")))
    check("archive: worker tree present",
          os.path.exists(os.path.join(WORK_ARCH, "models/org-alpha/hub/models--org--alpha")))
    wman = json.load(open(os.path.join(WORK_ARCH, "models/org-alpha/manifest.json")))
    check("archive: worker manifest v2 + node",
          wman.get("schema") == 2 and wman.get("node") == "worker", wman.get("node"))
    check("archive: worker manifest size matches its own payload",
          wman["payload"]["bytes"] == got["worker"]["bytes"], wman.get("payload"))

    rc, res = jrun("--list")
    rows = {m["id"]: m for m in res["models"]}
    check("list: alpha archived on both", rows["org/alpha"]["state"] == "archived", rows["org/alpha"])

    # --- unarchive: fit checked per node, restored on both
    rc, res = jrun("--unarchive", "org/alpha")
    check("unarchive: both restored",
          rc == 0 and res.get("total_restored_bytes") >= 65536
          and len(res.get("nodes", {})) == 2, res)
    check("unarchive: head files back",
          tree(os.path.join(head_hub, "models--org--alpha"))
          == tree(os.path.join(HEAD_ARCH, "models/org-alpha/hub/models--org--alpha")))
    check("unarchive: worker files back",
          os.path.exists(os.path.join(work_hub, "models--org--alpha/snapshots/abc/w2.safetensors")))

    # --- per-node space refusal names only the tight node
    node_conf(WORK_HOME, WORK_ARCH, min_free_gb=1000000)
    rc, res = jrun("--archive", "org/beta")
    check("beta archived", rc == 0, res)
    rc, res = jrun("--unarchive", "org/beta", expect=3)
    check("unarchive: refusal names the tight node only",
          rc == 3 and res.get("no_space_nodes") == ["worker"], res)
    check("unarchive: per-node shortfall",
          res.get("shortfall_nodes", {}).get("worker", 0) > 0, res.get("shortfall_nodes"))
    check("unarchive: head still fits", res["nodes"]["head"].get("fits") is True, res["nodes"]["head"])
    node_conf(WORK_HOME, WORK_ARCH, min_free_gb=10)
    rc, res = jrun("--unarchive", "org/beta")
    check("unarchive after margin fixed", rc == 0, res)

    # --- copy failure on the worker rolls the staged head copy back
    build(HEAD_HOME, {"org/gamma": (1, 4096)})
    build(WORK_HOME, {"org/gamma": (1, 8192)})
    node_conf(HEAD_HOME, HEAD_ARCH)
    node_conf(WORK_HOME, WORK_ARCH)
    # make the worker's copy stage fail deterministically: the archive dir
    # exists but "hub" is a regular file, so creating the tree raises
    os.makedirs(os.path.join(WORK_ARCH, "models/org-gamma"), exist_ok=True)
    write(os.path.join(WORK_ARCH, "models/org-gamma/hub"), b"not a dir")
    rc, res = jrun("--archive", "org/gamma", expect=4)
    check("rollback: error names the worker", "worker" in res.get("error", ""), res)
    check("rollback: head staged copy removed",
          not os.path.exists(os.path.join(HEAD_ARCH, "models/org-gamma")), res)
    check("rollback: both sources intact",
          os.path.isdir(os.path.join(head_hub, "models--org--gamma"))
          and os.path.isdir(os.path.join(work_hub, "models--org--gamma")))
    shutil.rmtree(os.path.join(WORK_ARCH, "models/org-gamma"), ignore_errors=True)

    # --- an unreachable node blocks the whole operation
    saved = os.environ.get("SPARK_VLLM_REMOTE_CMD")
    os.environ["SPARK_VLLM_REMOTE_CMD"] = "/bin/false {host} {helper} {args}"
    rc, res = jrun("--archive", "org/gamma", expect=5)
    if saved is None:
        os.environ.pop("SPARK_VLLM_REMOTE_CMD", None)
    else:
        os.environ["SPARK_VLLM_REMOTE_CMD"] = saved
    check("unreachable node: refused with a hint",
          rc == 5 and "not usable there" in res.get("error", "")
          and "deploy.sh" in (res.get("hint") or ""), res)
    check("unreachable node: nothing moved",
          os.path.isdir(os.path.join(head_hub, "models--org--gamma")))

    # --- unarchive --prune-archive drops both archive copies
    rc, res = jrun("--archive", "org/gamma")
    check("gamma archived on both", rc == 0, res)
    rc, res = jrun("--unarchive", "org/gamma", "--prune-archive")
    check("prune: both archive dirs gone",
          rc == 0 and not os.path.exists(os.path.join(HEAD_ARCH, "models/org-gamma"))
          and not os.path.exists(os.path.join(WORK_ARCH, "models/org-gamma")), res)
    check("prune: files back on both",
          os.path.isdir(os.path.join(head_hub, "models--org--gamma"))
          and os.path.isdir(os.path.join(work_hub, "models--org--gamma")))

    # --- human output carries one line per node
    r = run("--archive", "org/gamma", "--dry-run")
    check("human dry-run lists both nodes",
          "head" in r.stdout and "worker" in r.stdout and "total" in r.stdout, r.stdout)
    r = run("--status")
    check("human status lists both nodes", "head" in r.stdout and "worker" in r.stdout, r.stdout)
    r = run("--list")
    check("human list has a column per node",
          "HEAD" in r.stdout and "WORKER" in r.stdout, r.stdout[:300])

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
