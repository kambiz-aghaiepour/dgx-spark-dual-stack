# TensorFold stack (DeepSeek-V4.1-Flash-1017) — separate from the vLLM stack

This is a second, **mutually exclusive** serving stack for the pair: the TensorFold engine
(exllamav3-derived) instead of vLLM, port 8888 instead of 8000. It exists because the
DeepSeek-V4.1-Flash EXL3 packs need an EXL3 loader, and because TensorFold is measured at
~2× the vLLM kit's decode speed on this exact hardware (57 vs 27 tok/s prose, 2,179 vs
925 tok/s prefill at 32K).

Nothing here touches the vLLM stack (`miaai`/`b12x` engines, `spark-vllm-podman.service`,
`vllm-model-select`, the archive tool). The two stacks need the whole pair and ~100 GiB of
memory per node, so only one runs at a time.

## Layout

- `scripts/tensorfold-engine.sh` — build the engine image on the head, copy it to the worker, report both.
- `scripts/tensorfold-model.sh` — fetch a model view, start/stop/status/verify/logs.
- `config/tensorfold/cluster.env.example` — node addresses, RoCE devices, runtime (podman).
- `config/tensorfold/<profile>.env` — one file per model: `base` (LibertAI pack) and
  `dealign-direct` (the abliterated bundle).

The recipe itself is a fork with podman support:
`https://github.com/kambiz-aghaiepour/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark`,
branch `podman-support` (default checkout `~/ds41-recipe`).

## Setup once

```bash
mkdir -p ~/.config/spark-vllm/tensorfold
cp config/tensorfold/cluster.env.example ~/.config/spark-vllm/tensorfold/cluster.env
$EDITOR ~/.config/spark-vllm/tensorfold/cluster.env      # node IPs, WORKER_SSH, RoCE names, GID index

./scripts/tensorfold-engine.sh build dealign-direct       # podman build + copy to the worker
```

The recipe's host tuning runs once and is shared by both profiles:

```bash
cd ~/ds41-recipe && ./scripts/host_tuning.sh             # sudo on both nodes: swap off for the slice, THP off, earlyoom
```

## Fetch, start, stop

```bash
./scripts/tensorfold-model.sh fetch  dealign-direct   # ~416 GB download, view, worker copy, chat template
./scripts/tensorfold-model.sh doctor dealign-direct  # preflight only
./scripts/tensorfold-model.sh start  dealign-direct
./scripts/tensorfold-model.sh verify dealign-direct  # five real requests
./scripts/tensorfold-model.sh status dealign-direct
./scripts/tensorfold-model.sh stop   dealign-direct
```

`start` refuses while the vLLM stack is serving (`/v1/models` on 8000 or container `vllm-fn`),
and the recipe's own GPU guard refuses while any GPU container runs in either podman store.

## Space and memory

| | Per node |
|---|---|
| Weights (`dealign-direct`, 43 shards + two 101.5 GB Engram tables) | 416 GB |
| Rank cache + spill, first start | ~110 GB |
| Free now (internal NVMe) | 653 GB → fits, ~150 GB left |
| Memory needed free while it runs | ~100 GiB of 121 GiB |

`~/models` is on the internal NVMe. Do **not** put these weights on `/archive`: the engine
re-reads the Engram tables during prefill and the archive disk is a USB HDD.

## What was verified on the pair (2026-10-31)

`dealign-direct` with engine `6c53993796ef`: weights loaded in 273 s (head) / 355 s (worker),
103 GiB reserved per node, context 131072, 4 streams, DSpark 5 drafts, API on 8888.
`verify` 5/5 (plain, thinking, strict json_schema, tool_choice, regex). Decode measured
51.2 tok/s on a 123-token answer (the recipe's published figure is 57).

Two things cost time and are worth knowing:

- **The base engine revision cannot load this pack.** It demands `layers.N.attn.wo_b.trellis`;
  the bundle keeps those matrices in BF16 (`…wo_b.weight`), so the first start stopped with
  `KeyError: 'layers.10.attn.wo_b.trellis'` after the preflight passed. Revision `6c53993796ef`
  has the BF16 path (`is_bf16_only`), which is why the profile pins it. That revision is
  TensorFold PR #8, still unmerged upstream.
- **The health unit restarts vLLM.** `spark-vllm-health.service` brings `spark-vllm-podman`
  back within a minute of a stop, so taking the pair means stopping the health unit first.

## Which profile to try

1. **`dealign-direct`** — the abliterated bundle with the engine revision that is reported
   working for EXL3 on two Sparks. The bundle is the same format and shard count as the
   LibertAI pack, so this is the direct test. Expect MMLU to be ~4.2 pp below the base pack.
2. **`base`** — same stack, unmodified weights. Use it to prove the engine and the host
   tuning first, or to compare answers.
3. **BF16 overlay (not shipped here)** — the recipe branch can overlay 54 BF16 matrices from
   the bundle on the base pack (`scripts/uncensored.sh`). It needs TensorFold PR #8, which is
   unmerged, its head has not passed GPU qualification, and the saved checks do not prove
   every request fits in memory. Treat it as lab work until that lands.

### Network exposure

The profile sets `BIND_HOST=0.0.0.0`, so the engine answers on the head's LAN address
(`http://192.168.1.39:8888/v1`) as well as loopback, with **no API key** — anything on the home
network can use it. `tensorfold serve` accepts one `--host`/`--port` pair, so there is no way to
listen on 8000 as well; 8000 is deliberately left to the vLLM stack, which takes it when you
switch back. To close the engine again: `BIND_HOST=127.0.0.1` plus an `API_KEY_FILE`, then
`spark-stack-switch tensorfold --apply`.

## Switching stacks

`~/.config/spark-vllm/stack.conf` says which stack owns the pair (`STACK=vllm|tensorfold`,
`TF_PROFILE=…`); a missing file means vllm. Both selectors and both watchdogs read it, so
the two stacks never fight over the memory.

```bash
spark-stack-switch --status                  # what is written down vs what answers on 8000/8888
spark-stack-switch tensorfold --apply        # stop vLLM (+ its watchdog), start TensorFold (+ its watchdog)
spark-stack-switch vllm                      # stop TensorFold and its watchdog; vLLM starts on its own
```

In `vllm-model-select` the stack is a row of its own, always first:
`tensorfold:dealign-direct  tf  running/stopped`. Enter picks it, `A` (or `s`) applies it —
exactly like a model row, except applying it runs the switcher instead of `vllm-model-set`.
Archive, unarchive, lock and delete refuse on that row: the weights live in `~/models`, not
in the HF cache, so there is nothing there to move or delete. Choosing a vLLM model while
TensorFold runs makes `vllm-model-set` stop TensorFold first.

Units (installed by `deploy.sh`, enabled, never started by it):

- `spark-tensorfold.service` — oneshot + `RemainAfterExit`; ExecStart is a no-op unless
  `stack.conf` says tensorfold, which is what makes it safe to enable at boot.
- `spark-tensorfold-watchdog.service` — probes `/v1/models` and a 1-token completion on 8888,
  restarts the unit after 10 consecutive failures, and idles whenever `stack.conf` says vllm.
  A stack started by hand (without the unit) is left alone while it still answers.

`spark-vllm-watchdog` gained the mirror-image guard: while `stack.conf` says tensorfold it
sleeps instead of trying to restart a stack that is supposed to be down.

## Interaction with the rest of the repo

- `vllm-model-select` / `vllm-model-archive` only know `~/.cache/huggingface/hub`, so these
  views are invisible to them — no archiving, no locking, no `x`/`u`. Leave them alone.
- `deploy.sh` does not install these launchers: they are run from this checkout, because they
  resolve `config/tensorfold/` relative to the repo.
- To go back to the vLLM stack: `./scripts/tensorfold-model.sh stop`, then
  `systemctl --user start spark-vllm-podman && systemctl --user start spark-vllm-health`
  (the health unit stays stopped while TensorFold runs, otherwise it restarts vLLM).
