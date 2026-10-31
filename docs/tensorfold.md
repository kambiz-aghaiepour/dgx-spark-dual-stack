# TensorFold stack (MiaAI-Lab DeepSeek-V4.1-Flash-1017)

The second serving stack: a **mutually exclusive** alternative to vLLM, run instead of it
and never beside it. It serves DeepSeek-V4.1-Flash-1017 with TensorFold's native Zig engine
(`tensorfold-native`, BertholomusAI's `dsv41` family) from the pinned MiaAI-Lab recipe
[DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks).
The recipe itself is third-party (Apache-2.0) and lives outside this repo, at
`~/mia-dsv41` by default; `scripts/setup-mia-recipe.sh` clones the pin and installs its
`scripts/local.sh`.

## What it serves

- Model: `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` (EXL3, ~196 GiB) plus the
  base model's unquantized Engram tables (shards 47/48, ~203 GB).
- Served name: `DeepSeek-V4.1-Flash-1017`; API on port **8000** (both stacks share it; only
  one runs at a time).
- Context 1,048,576 tokens, 4 concurrent streams, DSpark drafts, vision, tool calls.
- No API key by default (`API_KEY=0`); the port is bound to 0.0.0.0.

## Prerequisites

- Docker with GPU support on both Sparks, and the login user in the `docker` group
  (`sudo usermod -aG docker kambiz` on both; log in again or use a new ssh session).
- The Hugging Face CLI (`hf`) on the head. `prepare.sh` downloads ~400 GB.
- An NFS export of the head's `HF_CACHE` to the worker's CX7 address, because the worker
  reads the checkpoint over NFS:
  `/archive/hf 192.168.177.12(rw,sync,no_subtree_check,no_root_squash)` in `/etc/exports`,
  then `sudo exportfs -ra && sudo systemctl enable --now nfs-server`.
- ~500 GB free on `/archive` on each node; the worker's Engram copy and rank cache live on
  its own `/archive`.

## Space layout

| Path | What |
|---|---|
| `/archive/hf` (head) | HF cache: checkpoint, Engram shards, kit, rank 0 cache |
| `/archive/mia/engram` (worker) | Engram tables, local copy (fast random reads) |
| `/archive/mia/rank-cache` (worker) | rank 1 weight file (~99 GiB) |

## First setup

```bash
./scripts/setup-mia-recipe.sh                # clone ~/mia-dsv41 at the pinned commit
$EDITOR ~/mia-dsv41/scripts/local.sh          # WORKER, HF_CACHE, WORKER_* dirs, PORT, CONTEXT
cd ~/mia-dsv41 && ./start.sh                  # build image, download weights, start both ranks
cd ~/mia-dsv41 && ./verify.sh                 # five real requests against the API
```

`start.sh` auto-runs `prepare.sh` when the image, checkpoint, weight files or worker copies
are missing; `DRY_RUN=1 ./start.sh` prints both ranks' docker commands without changing
anything.

## Switching

```bash
spark-stack-switch --status
spark-stack-switch tensorfold --apply   # stop vLLM (+ watchdog), start TensorFold (+ watchdog)
spark-stack-switch vllm                 # stop TensorFold (+ watchdog); vLLM starts on its own
```

The systemd user units are `spark-tensorfold.service` (oneshot, `ExecStart` is a no-op
unless `~/.config/spark-vllm/stack.conf` says `STACK=tensorfold`) and
`spark-tensorfold-watchdog.service` (probes `/v1/models` and a one-token completion on
port 8000; restarts the unit after 10 consecutive failures; idle under the vLLM stack).

## Tuning

Edit `~/mia-dsv41/scripts/local.sh` (layer: environment > local.sh > `.env` > defaults).
Useful keys: `CONTEXT` (default 1048576), `PARALLEL`, `DRAFTS`, `PORT`, `TOP_P`, `TOP_K`,
`THINKING`, `SPILL_GIB` (off by default; if enabled, also set `HEAD_ENGRAM_DIR` to a local
folder with the Engram shards, or the rank fingerprint refuses to load). Re-run
`./start.sh restart` after a change.

## Notes

- The recipe writes no state into this repo. `spark-stack-switch`, the selector row
  (`tensorfold:mia-dsv41`) and the watchdog are the integration surface.
- The vLLM models and their archive are unchanged: `vllm-model-select` still lists, switches
  and archives them, and `stack.conf= vllm` returns the pair to that stack.
