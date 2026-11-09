# TensorFold stack (MiaAI-Lab recipes)

The second serving stack: a **mutually exclusive** alternative to vLLM, run instead of it
and never beside it. TensorFold is the engine; a **profile** is the recipe checkout plus the
checkpoint it serves, and this repo ships two:

- `mia-dsv41` — **DeepSeek-V4.1-Flash-1017**, from the pinned MiaAI-Lab recipe
  [DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks),
  checkout `~/mia-dsv41`.
- `glm53` — **GLM-5.3-Flash-EXL3** (gated Ablit weights), from
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold),
  checkout `~/glm53`.

One systemd unit, `spark-tensorfold.service`, runs whichever profile `stack.conf` names, so the
live stack is also the one that starts after a reboot. Both recipes are third-party and live
outside this repo; both pins are **our forks** that add the container-runtime switch this pair
runs on (`CONTAINER_RT`, `scripts/runtime.sh`), offered upstream as
[MiaAI-Lab#113](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/113) and
[MiaAI-Lab#246](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/246).
Both forks keep upstream's licence; docker remains the upstream default.

## What it serves

Both profiles share port **8000** (only one runs at a time) and neither uses an API key by
default (`API_KEY=0`; bound to 0.0.0.0 — LAN exposure is deliberate).

| | `mia-dsv41` | `glm53` |
|---|---|---|
| Model | `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` + unquantized Engram tables (shards 47/48) | `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit` (gated) + `incoai/GLM-5.3-Flash-DFlash2` drafts |
| Served name | `DeepSeek-V4.1-Flash-1017` | `GLM-5.3-Flash-EXL3` |
| Checkout | `~/mia-dsv41` | `~/glm53` |
| Weights | `/srv/mia/hf`, worker reads over NFS | `/srv/mia/glm53`, local copy on each node |
| Container | `dsv41-1017-exl3-tf` | `glm53-flash-tf` |
| Footprint | ~200 GB + ~110 GB rank caches per node | ~180 GB per node |
| Context / streams | 1,048,576 · 4 · DSpark drafts · vision | **262,144** · 4 · DFlash2 drafter · vision |
| Measured | 55.3–55.6 tok/s single stream | 41.6 tok/s single stream, 92.4 tok/s at 4 |

GLM's Ablit weights are gated: the account has to accept the terms on the model page, and
`start.sh` refuses `ABLIT=1` without `HF_TOKEN` even when the weights are already on disk.
`scripts/setup-glm-recipe.sh` writes the token to `~/.config/spark/glm53.env` (0600) — outside
the checkout and outside the unit file — and the start wrapper loads it for that profile only.

## Prerequisites

- **Podman** (4.9+, for `--gpus all`) on both Sparks, driven **rootful** through passwordless
  `sudo -n podman`: the worker's read-only NFS volume and the `tensorfold.slice` cgroup parent
  both need root. Docker still works (`CONTAINER_RT=docker` in `scripts/local.sh`); the recipe
  documents both in `docs/PODMAN.md` of its own tree.
- The Hugging Face CLI (`hf`) on the head. `prepare.sh` downloads ~400 GB.
- An NFS export of the head's `HF_CACHE` to the worker's CX7 address, because the worker
  reads the checkpoint over NFS:
  `/srv/mia/hf 192.168.177.12(rw,sync,no_subtree_check,no_root_squash)` in `/etc/exports`,
  then `sudo exportfs -ra && sudo systemctl enable --now nfs-server`.
- ~500 GB free on `/archive` on each node; the worker's Engram copy and rank cache live on
  its own `/archive`.

## Space layout

| Path | What |
|---|---|
| `/srv/mia/hf` (head) | `mia-dsv41`: HF cache — checkpoint, Engram shards, kit, rank 0 cache |
| `/srv/mia/engram` (worker) | `mia-dsv41`: Engram tables, local copy (fast random reads) |
| `/srv/mia/rank-cache` (worker) | `mia-dsv41`: rank 1 weight file (~99 GiB) |
| `/srv/mia/glm53` (both) | `glm53`: the gated checkpoint and drafter, one local copy per node |
| `~/.cache/tensorfold-glm53/<image>` (both) | `glm53`: compiled CUDA kernels, one folder per image hash |

All of these are on each node's internal NVMe. `/archive` is the external USB volume and is
reserved for archived vLLM models; no running engine reads from it.

## First setup

DeepSeek profile (the `mia-dsv41` one):

```bash
./scripts/setup-mia-recipe.sh                # clone ~/mia-dsv41 at the pinned commit
$EDITOR ~/mia-dsv41/scripts/local.sh          # WORKER, HF_CACHE, WORKER_* dirs, PORT, CONTEXT
cd ~/mia-dsv41 && ./start.sh                  # build image, download weights, start both ranks
cd ~/mia-dsv41 && ./verify.sh                 # five real requests against the API
```

GLM profile (`glm53`; accept the gate on the model page first):

```bash
./scripts/setup-glm-recipe.sh                # clone ~/glm53 at the pinned commit, write local.sh + token file
$EDITOR ~/glm53/scripts/local.sh              # WORKER, HF_CACHE=/srv/mia/glm53, WORKER_WEIGHTS=copy
cd ~/glm53 && ./scripts/prepare.sh            # pinned image from GHCR + the ~180 GB gated checkpoint + worker copy
spark-stack-switch tensorfold glm53 --apply   # first start compiles the CUDA kernels (~6 min), then serves
```

`start.sh` auto-runs `prepare.sh` when the image, checkpoint, weight files or worker copies
are missing; `DRY_RUN=1 ./start.sh` prints both ranks' runtime commands without changing
anything. The official control path is `spark-stack-switch` (it starts/stops the systemd
unit); `./start.sh` is the recipe's implementation detail and is what the unit runs.

## Runtime

The pair runs the recipe under **rootful podman**; `~/mia-dsv41/scripts/local.sh` carries:

```sh
CONTAINER_RT=podman
CONTAINER_RT_SUDO=1
CGROUP_PARENT=tensorfold.slice     # the host tuning's slice: MemorySwapMax=0
```

Podman keeps its own image store, so the image is copied once per node when moving from
Docker: `sudo -n docker save tensorfold-dsv41:<tag> | sudo -n podman load` on the head, and
the same through `ssh kambiz@gx10-02` on the worker. `scripts/prepare.sh` then sees the image
missing from the podman store and copies it from the head if it is not there. Rollback is
`CONTAINER_RT=docker` (plus `./start.sh restart`): the Docker image and container name are
untouched by a podman run.

```bash
sudo -n podman ps --format '{{.Names}} {{.Status}}'      # dsv41-1017-exl3-tf or glm53-flash-tf, on both nodes
sudo -n podman logs -f dsv41-1017-exl3-tf                # rank 0 (ssh gx10-02 for rank 1)
sudo -n podman logs -f glm53-flash-tf                    # the GLM profile
```

### GLM memory: 262,144 context, not 1M

A 121 GiB node holds ~89 GiB of GLM weights, so the shared KV pool lives in what is left. At the recipe's default
`CONTEXT=1048576` with `PARALLEL=4`, four days of 60-95k-token conversations filled it: earlyoom on the worker
(`gx10-c7ae`) sent **113 SIGTERM**s to the rank and finally SIGKILL, the kernel logged
`NVRM: ... Out of memory [NV_ERR_NO_MEMORY]`, rank 1 died, rank 0 stopped with it, and the pair kept answering
`/v1/models` from rank 0 alone while every real request blocked - an engine that looks alive and serves nothing.

`~/glm53/scripts/local.sh` sets:

```sh
CONTEXT=262144        # 2.7x the largest conversation seen; 4x less KV than 1M
MEMORY_RESERVE_GIB=20 # host headroom above earlyoom's line (recipe default is 14.5 at PARALLEL=4)
```

Measured after the change: a cold 94,988-token prompt returns in **47 s** (2,021 tok/s prefill), the identical request
repeats in **0.2 s** from the prefix cache, and available memory stays flat at 15 GiB (head) / 17 GiB (worker) before
and after. Watch `free -g` on **both** nodes and `journalctl | grep earlyoom` on the worker: the head is not the one
that dies first.

The GLM recipe refuses a bind mount whose source does not exist (docker used to create it), so
`start.sh` makes `~/.cache/tensorfold-glm53/<image>` on both nodes before launching; and under
rootful podman the gated download needs `HF_TOKEN` passed by value, because `sudo` resets the
environment a bare `-e HF_TOKEN` would read. Both are fixed in the fork (PR #246).

## Switching

```bash
spark-stack-switch --status                 # stack.conf vs what answers, and what boots
spark-stack-switch tensorfold glm53 --apply # stop vLLM, run the GLM profile (+ watchdog)
spark-stack-switch tensorfold mia-dsv41     # back to DeepSeek (needs its payload unarchived)
spark-stack-switch tensorfold mia-dsv41 --apply
spark-stack-switch vllm --apply             # stop TensorFold (+ watchdog), start vLLM
```

**The live stack is the boot default.** Every switch enables the incoming stack's units at boot
and disables the outgoing stack's, so a reboot comes up on exactly what was running; there is no
second enable flag that can drift. `verify-deployment.sh` fails if `stack.conf` and the unit
enablement disagree.

The systemd user units are `spark-tensorfold.service` (oneshot; `ExecStart` runs the profile's
recipe, and is a no-op unless `~/.config/spark-vllm/stack.conf` says `STACK=tensorfold`) and
`spark-tensorfold-watchdog.service` (probes `/v1/models` and a one-token completion against the
live profile's model id on port 8000; restarts the unit after 10 consecutive failures; idle
under the vLLM stack).

Profiles are defined in `scripts/tensorfold-profiles.sh` (directory, model id, container, image,
token file). Adding a third recipe means adding one `case` arm there and one in
`spark-stack-switch`'s usage text — the unit, watchdog and verify checks follow it.

`stack.conf` also records `VLLM_MODEL`, the parked vLLM choice. It is only the default
while `STACK=vllm`. Under TensorFold it can be archived; `spark-stack-switch vllm` checks
both nodes and refuses to stop TensorFold until that checkpoint is restored to internal
NVMe. Use `vllm-model-archive.py --unarchive <id>` (or `u` in the selector).

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
