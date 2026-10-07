#!/usr/bin/env python3
"""vllm-model-add - Phase 3: add a model from Hugging Face to the DGX cluster.

Validates (exists / gated / size / fit vs 121.6 GiB per node / arch support),
creates a per-family flags template in ~/.config/spark-vllm/flags/, downloads
to the head and rsync-copies to the worker via the repo's hf-download.sh.

Usage: vllm-model-add <hf-repo-or-url> [--alias NAME] [--force]
"""
import json
import os
import subprocess
import sys
import urllib.request

CONF_DIR = os.path.expanduser("~/.config/spark-vllm")
FLAGS_DIR = os.path.join(CONF_DIR, "flags")
TOKEN_FILE = os.path.expanduser("~/.cache/huggingface/token")
REPO_DIR = os.path.expanduser("~/spark-vllm-docker")
GB10_TOTAL_GIB = 121.6
# Measured usable-before-engine budget on GB10 (unified 128GB minus ~20GiB
# reserved): both Qwen (100.87 free) and MiniMax (100.22 free) hit this wall.
GB10_BUDGET_GIB = 100.9
TP_SIZE = 2

KNOWN_FAMILY_FLAGS = {
    "deepseek_v4": "deepseek-v4.conf",   # existing deepseek flag set
}

GENERIC_FLAGS = [
    "--host", "0.0.0.0", "--port", "8000", "--trust-remote-code",
    "--tensor-parallel-size", "2", "--kv-cache-dtype", "fp8",
    "--block-size", "256", "--max-model-len", "auto",
    "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
    "--gpu-memory-utilization", "0.83", "--enable-prefix-caching",
    "--max-cudagraph-capture-size", "48",
]

QWEN_FLAGS = [("0.82" if f == "0.83" else f) for f in GENERIC_FLAGS]


def hf(rid, path, raw=False):
    tok = open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else ""
    req = urllib.request.Request(
        f"https://huggingface.co/{rid}/resolve/main/{path}",
        headers={"Authorization": f"Bearer {tok}"} if tok else {})
    data = urllib.request.urlopen(req, timeout=40).read()
    return data if raw else json.loads(data)


def repo_size_gb(rid):
    d = json.load(urllib.request.urlopen(
        f"https://huggingface.co/api/models/{rid}/tree/main?recursive=true",
        timeout=40))
    return sum(f.get("size", 0) for f in d
               if f.get("type") == "file" and f["path"].endswith(".safetensors")) / 1e9


def registry_archs():
    out = subprocess.run(
        ["podman", "exec", "vllm_node", "python3", "-c",
         "from vllm.model_executor.models.registry import ModelRegistry as R;"
         "print('\\n'.join(R.get_supported_archs()))"],
        capture_output=True, text=True, timeout=60).stdout
    return set(out.split())


def arch_supported(cfg, archs):
    cand = cfg.get("architectures") or []
    return [a for a in cand if a in archs] or None


def parse_id(arg):
    arg = arg.strip()
    if arg.startswith("http"):
        arg = arg.rstrip("/").split("huggingface.co/")[-1]
    arg = arg.strip("/")
    assert "/" in arg and " " not in arg, f"not a HF repo id: {arg}"
    return arg


def family_flags(model_type, rid):
    if model_type in KNOWN_FAMILY_FLAGS:
        return os.path.join(FLAGS_DIR, KNOWN_FAMILY_FLAGS[model_type])
    safe = rid.replace("/", "-").replace(".", "_")[:60] + ".conf"
    lines = QWEN_FLAGS if model_type.startswith("qwen") else GENERIC_FLAGS
    path = os.path.join(FLAGS_DIR, safe)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    rid = parse_id(args[0])
    alias = None
    force = "--force" in sys.argv
    for a in sys.argv[1:]:
        if a.startswith("--alias"):
            alias = sys.argv[sys.argv.index(a) + 1]
    # 1) config + arch check (pre-download)
    cfg = hf(rid, "config.json")
    mtype = cfg.get("model_type", "?")
    archs = registry_archs()
    sup = arch_supported(cfg, archs)
    print(f"repo={rid} model_type={mtype} architectures={cfg.get('architectures')}")
    if sup:
        print(f"arch supported: {' '.join(sup)}")
    elif not force:
        print(f"ERROR: no architecture of {cfg.get('architectures')} in engine registry; "
              f"use --force to proceed anyway")
        sys.exit(2)
    # 2) size + fit (weights scale linearly-ish with TP; GB10 free-mem ceiling ~100.9 GiB)
    size = repo_size_gb(rid)
    per_node_gib = size / TP_SIZE / 1.0737
    print(f"weights: {size:.1f} GB total -> {per_node_gib:.1f} GiB/node (TP2)")
    if per_node_gib > GB10_BUDGET_GIB and not force:
        print(f"ERROR: {per_node_gib:.1f} GiB/node exceeds measured GB10 budget "
              f"({GB10_BUDGET_GIB} GiB, unified RAM minus ~20 GiB reserved); "
              f"use --force to try anyway")
        sys.exit(2)
    # 3) flags template
    flags = family_flags(mtype, rid)
    print(f"flags template: {flags}")
    # 4) download + worker copy (existing proven path)
    tok = open(TOKEN_FILE).read().strip() if os.path.exists(TOKEN_FILE) else ""
    env = dict(os.environ, HF_TOKEN=tok,
               PATH=os.path.expanduser("~/.local/bin") + os.pathsep +
               os.environ.get("PATH", ""))
    print(f"downloading {rid} (head) + rsync to worker...")
    r = subprocess.run(
        ["./hf-download.sh", rid, "-c"], cwd=REPO_DIR, env=env, timeout=None)
    print("download/copy rc:", r.returncode)
    if r.returncode != 0:
        sys.exit(3)
    print(f"\nready. flags: {flags}\n"
          f"switch with: vllm-model-set {rid} {alias or rid} --apply")


if __name__ == "__main__":
    main()
