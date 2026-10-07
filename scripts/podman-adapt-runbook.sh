#!/bin/bash
# podman-adapt-runbook.sh — clone the MiaAI Qwen3.8-Flash-Next runbook and
# adapt it to this host (rootless podman, port 8000, our CX7 fabric).
#
# Run once per fresh install (before the first flash-next switch).
# Upstream (AGPL-3.0, credit/reference in README):
#   https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks
set -euo pipefail
RUNBOOK="${MIAI_RUNBOOK_DIR:-$HOME/miaai-qwen38}"
REPO_URL="${MIAI_RUNBOOK_REPO:-https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks}"

HEAD_IP="${HEAD_IP:-192.168.177.11}"
WORKER_IP="${WORKER_IP:-192.168.177.12}"
IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-=rocep1s0f1}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"

if [ ! -d "$RUNBOOK" ]; then
  echo "==> cloning runbook: $REPO_URL"
  git clone --depth 1 "$REPO_URL" "$RUNBOOK"
fi

echo "==> podman adaptation (docker -> podman in start/stop scripts)"
cd "$RUNBOOK"
for f in start.sh stop.sh start-fp8.sh start-v030.sh; do
  [ -f "$f" ] && sed -i 's/\bdocker\b/podman/g' "$f" && echo "    adapted: $f"
done

echo "==> fabric .env (port 8000, our mesh)"
cat > "$RUNBOOK/.env" <<EOF
HEAD_IP="$HEAD_IP"
WORKER_IP="$WORKER_IP"
WORKER_USER=""
IFACE="$IFACE"
IB_HCA="$IB_HCA"
IB_GID_INDEX=$IB_GID_INDEX
MASTER_PORT=50000
PORT=8000
MODEL_ID=nvidia/Qwen3.8-Flash-Next-NVFP4
SERVED_MODEL_NAME=nvidia/Qwen3.8-Flash-Next-NVFP4
IMAGE="vllm/vllm-openai:qwen38-flash-next"
KV_CACHE_DTYPE=fp8
MAMBA_SSM_CACHE_DTYPE=bfloat16
TENSOR_PARALLEL_SIZE=2
ENABLE_EXPERT_PARALLEL=true
MM_ENCODER_TP_MODE=data
MTP_NUM_SPECULATIVE_TOKENS=3
MTP_DISABLE_BLOCK_DROP=1
MTP_INDEX_SHARE=true
MTP_DRAFT_VOCAB="files/draft_vocab_en_code_47k.txt"
QSA_PROFILE=stock
PLE_OFFLOAD=false
REQUIRE_IDLE_GPU=true
EVICT_PAGE_CACHE=true
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
NFS_SHARE=false
EOF
echo "==> done. Image is pulled on first launch (warm start uses --launch)."
