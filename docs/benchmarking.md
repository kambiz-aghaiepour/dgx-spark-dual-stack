# Benchmarking

## Protocol

Single steady-state client, **identical row for every comparison**:

```bash
llama-benchy --base-url http://<head-ip>:8000/v1 \
  --model <served-model-id> --tokenizer <served-model-id> \
  --pp 1024 --tg 800 --depth 0 8192 --runs 3 \
  --latency-mode generation --skip-coherence
```

- Used in every baseline/tuned/native-stack measurement (no cherry-picked rows).
- `--latency-mode generation` + `--skip-coherence` for stable single-stream
  numbers.
- pp@d0 can look inflated (e.g. 49k t/s) when the warmup prompt hits the
  prefix cache — trust the `@ d8192` row for prefill.

## Results on this kit (2026-10-07)

| Config | Decode t/s (pp1024/tg800) | Peak | @8K depth | TTFT @8K |
|---|---|---|---|---|
| b12x plain | 20.03 ± 0.46 | — | 20.43 | ~3.1 s |
| b12x tuned (MTP 3 + EP + FP8 KV) | 26.14 | — | 26.70 | ~3.1 s |
| **MiaAI native stack** | **45.51 ± 6.34** | **68.33** | **37.80** | **3.16 s** |

Native stack recipe: `vllm/vllm-openai:qwen38-flash-next` + overlay patches,
TP=2, MTP **3** speculative tokens, expert parallel on, FP8 KV cache,
`--mm-encoder-tp-mode data`, 47k draft vocabulary, QSA profile `stock`.

## Decision rule used (dual-stack integration)

- **≥40 t/s** on the native stack → proceed with the dual-stack wrapper
  (recorded outcome: **45.5 → integrated**).
- **<40 t/s** → report measurement + delta analysis and ask before wiring it in.

## Reproducing

```bash
cd tools && python3 sustained-tps-test.py --base-url http://127.0.0.1:8000/v1 ...
```

(optional; keep the box to one benchmark client at a time — the engine saturates
quickly and concurrent clients muddy single-stream numbers.)
