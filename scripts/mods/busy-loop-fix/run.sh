#!/bin/bash
# busy-loop-fix: GB10 thermal mitigation for vLLM tensor-parallel spin-wait.
# vLLM's inter-process broadcast defaults busy_loop_s=1 (up to 1s busy-spin
# during decode), pinning 3-4 X925 P-cores at 3.9 GHz on the shared CPU/GPU
# package. Lower to 0.002s. See gx10-thermal-guidance.md.
set -e
PY=/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/shm_broadcast.py
test -f "$PY" || { echo "busy-loop-fix: $PY not found" >&2; exit 1; }
cp -n "$PY" "$PY.bak" || true
sed -i -E 's/^([[:space:]]*busy_loop_s: float = )[0-9.]+/\10.002/' "$PY"
grep -n 'busy_loop_s: float' "$PY"
echo "busy-loop-fix: applied"
