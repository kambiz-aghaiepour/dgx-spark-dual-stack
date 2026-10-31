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

## Interaction with the rest of the repo

- `vllm-model-select` / `vllm-model-archive` only know `~/.cache/huggingface/hub`, so these
  views are invisible to them — no archiving, no locking, no `x`/`u`. Leave them alone.
- `deploy.sh` does not install these launchers: they are run from this checkout, because they
  resolve `config/tensorfold/` relative to the repo.
- To go back to the vLLM stack: `./scripts/tensorfold-model.sh stop`, then
  `systemctl --user start spark-vllm-podman`.
