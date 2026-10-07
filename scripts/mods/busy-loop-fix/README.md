# busy-loop-fix

Patches vLLM `busy_loop_s` from `1` to `0.002` in
`vllm/distributed/device_communicators/shm_broadcast.py` (both cluster nodes
via `--apply-mod` at launch).

Why: with TP >= 2 vLLM's inter-process queue busy-spins up to 1 s during
decode, holding 3-4 P-cores at 3.9 GHz on the shared CPU/GPU package and
pushing SoC temperature into the worry band. Community reports show CPU time
dropping from ~333% to ~89% with identical throughput/latency.

Re-applied automatically on every `spark-vllm-podman` start (see
`~/.local/bin/spark-vllm-podman-start`). Backups left as `shm_broadcast.py.bak`.
