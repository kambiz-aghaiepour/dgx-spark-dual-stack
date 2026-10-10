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

## Concurrency & workload-type sweep (flash-next native stack, 2026-10-07)

Harness: `tools/stack-sweep.py` (concurrent OpenAI-API clients, greedy,
requests kept in flight for 50 s; aggregate from `usage` fields).

| Workload | C | Aggregate tok/s | Per-stream p50 | p95 |
|---|---|---|---|---|
| Chat 1.2K→1024 (decode-heavy) | 1 | 61 | 50.3 | 56.7 |
| | 2 | 123 | 47.1 | 49.0 |
| | 4 | 164 | 37.0 | 42.3 |
| | 8 | 328 | 27.9 | 30.8 |
| | 12 | 410 | 25.8 | 32.4 |
| | 16 | 492 | 14.9 | 32.5 |
| Long ctx 16K→256 | 4 | 105 | 26.6 | 29.5 |
| Long ctx 64K→128 | 4 | 78 | 21.5 | 25.8 |
| Vision (512px PNG→describe) | 2 | 84 | 38.6 | 42.2 |
| Reasoning-heavy (512 out incl. reasoning) | 4 | 157 | 36.7 | 40.5 |

Notes:
- Decode-heavy row uses a repetitive numbers-listing prompt (chain-friendly for
  MTP-3 spec decode) — **realistic diverse text is ~45 t/s single-stream**
  (llama-benchy row above), so expect ~0.75–0.9× these aggregate figures for
  real prose.
- `MAX_NUM_SEQS=8`: beyond C=8 the extra requests queue; the tail (p50) grows
  at C=16 while aggregate keeps climbing (no errors observed up to 16).
- Reference (b12x/DeepSeek stack, prior measurement): peak **~72.7 tok/s API
  aggregate** (64K ctx, 16 users) — the native stack delivers ~6.8× that at
  the same box.

## Kernel A/B: 6.17.0-1032 vs 7.0.0-1019 (2026-10-10)

Same hardware, same native stack, same recipe — the running kernel is the only
variable. Both kernels are Ubuntu `-nvidia` flavor (DGX OS, Ubuntu 24.04 noble);
driver 580.178.04 / CUDA 13.0 on both; 6.17 required the per-kernel
`linux-modules-nvidia-580-open-*` package (the kernel package alone does NOT
carry the NVIDIA modules — first boot without it fails `nvidia-smi`).

| Metric (stock Qwen3.8-Flash-Next-NVFP4) | 7.0.0-1019 | 6.17.0-1032 | Δ 6.17 |
|---|---|---|---|
| Decode @ d0 | 45.51 ± 6.34 (peak 68.3) | 42.00 ± 2.34 (peak 69.7) | −7.7% mean / +2% peak |
| Decode @ 8K | 37.80 ± 1.93 (peak 57.7) | 38.33 ± 0.36 (peak 62.0) | +1.4% mean / +7.5% peak |
| Prefill @d0 / @8K (TTFT) | 2871 / 3600 (3.16 s) | 2871 / 3170 (3.07 s) | ≈ / −3% TTFT |

**Conclusion: the claimed "7.x performance hit" did not reproduce.** The d0
delta is inside the 7.0 baseline's own ±6.34 run variance; at depth 6.17 is
marginally ahead. Both are single bench sessions (runs=3 each).

## Uncensored vs stock on the same kernel (6.17, 2026-10-10)

| Metric | stock | keys (uncensored) | Δ keys |
|---|---|---|---|
| Decode @ d0 | 42.00 ± 2.34 (peak 69.7) | 40.78 ± 3.60 (peak 63.0) | −2.9% mean / −9.6% peak |
| Decode @ 8K | 38.33 ± 0.36 (peak 62.0) | 34.50 ± 3.24 (peak 50.0)* | −10% (run variance 34.5↔40.8) |
| Prefill | ≈ | ≈ / lower @ d0 | mixed |

*run-to-run spread at depth for the keys checkpoint is wide: 34.5 vs 40.8
between identical runs. Direction matches "uncensored is a bit slower" (small,
noise-level at shallow depth); quality impacts (abliteration) are not measured
by this bench.

## Reproducing

```bash
cd tools && python3 sustained-tps-test.py --base-url http://127.0.0.1:8000/v1 ...
```

(optional; keep the box to one benchmark client at a time — the engine saturates
quickly and concurrent clients muddy single-stream numbers.)
