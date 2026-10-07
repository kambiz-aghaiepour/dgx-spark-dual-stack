#!/bin/bash
# Dual-stack branch for nvidia/Qwen3.8-Flash-Next-NVFP4: the model-native
# engine (MiaAI-Lab stock-vLLM image `vllm/vllm-openai:qwen38-flash-next`
# + overlay patches), podman-adapted per dual-stack-integration-plan.
# Renders the runbook's .env from model.conf + per-model env + our fabric
# constants, then runs their launcher with `--launch` (skip download/sync;
# weights + image already on both nodes).
set -euo pipefail
export CONTAINER_RT=podman
CONF="${SPARK_VLLM_MODEL_CONF:-$HOME/.config/spark-vllm/model.conf}"
[ -f "$CONF" ] || { echo "miaai_launch: model.conf not found: $CONF"; exit 1; }
. "$CONF"
FLAGS_FILE="${FLAGS_FILE:-$HOME/.config/spark-vllm/flags/nvidia-Qwen3_8-Flash-Next-NVFP4.conf}"
ENVS="${FLAGS_FILE%.conf}.env"

RUNBOOK="${MIAI_RUNBOOK_DIR:-$HOME/miaai-qwen38}"
[ -x "$RUNBOOK/start.sh" ] || { echo "miaai_launch: runbook not found: $RUNBOOK/start.sh"; exit 1; }

# --- Fabric constants (this cluster) ---
HEAD_IP="${HEAD_IP:-192.168.177.11}"
WORKER_IP="${WORKER_IP:-192.168.177.12}"
WORKER_USER="${WORKER_USER:-}"
IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-=rocep1s0f1}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"
MASTER_PORT="${MASTER_PORT:-50000}"
PORT="${PORT:-8000}"
REQUIRE_IDLE_GPU="${REQUIRE_IDLE_GPU:-true}"
EVICT_PAGE_CACHE="${EVICT_PAGE_CACHE:-true}"
IMAGE="vllm/vllm-openai:qwen38-flash-next"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_ID}"

# --- Measured Step-A recipe (45.5 t/s mean, gate passed) ---
# Any of these can be overridden per model via the flags-file .env.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-bfloat16}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-true}"
MM_ENCODER_TP_MODE="${MM_ENCODER_TP_MODE:-data}"
MTP_NUM_SPECULATIVE_TOKENS="${MTP_NUM_SPECULATIVE_TOKENS:-3}"
MTP_DISABLE_BLOCK_DROP="${MTP_DISABLE_BLOCK_DROP:-1}"
MTP_INDEX_SHARE="${MTP_INDEX_SHARE:-true}"
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-files/draft_vocab_en_code_47k.txt}"
QSA_PROFILE="${QSA_PROFILE:-stock}"
ABLIT="${ABLIT:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
YARN_ENABLE=false
YARN_FACTOR="${YARN_FACTOR:-4.0}"

# Per-model env passthrough (their renderer only reads the names above;
# unknown vars are inert, so b12x-only knobs can safely ride along in the
# same per-model env file).
if [ -f "$ENVS" ]; then
  while IFS= read -r line; do
    case "$line" in ''|\#*) continue ;; esac
    export "$line"
  done < "$ENVS"
fi

cat > "$RUNBOOK/.env" <<EOF
HEAD_IP="$HEAD_IP"
WORKER_IP="$WORKER_IP"
WORKER_USER="$WORKER_USER"
IFACE="$IFACE"
IB_HCA="$IB_HCA"
IB_GID_INDEX=$IB_GID_INDEX
MASTER_PORT=$MASTER_PORT
PORT=$PORT
MODEL_ID="$MODEL_ID"
ABLIT=$ABLIT
SERVED_MODEL_NAME="$SERVED_MODEL_NAME"
MAX_MODEL_LEN=$MAX_MODEL_LEN
YARN_ENABLE=$YARN_ENABLE
YARN_FACTOR=$YARN_FACTOR
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_NUM_SEQS=$MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
KV_CACHE_DTYPE=$KV_CACHE_DTYPE
MAMBA_SSM_CACHE_DTYPE=$MAMBA_SSM_CACHE_DTYPE
TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE
ENABLE_EXPERT_PARALLEL=$ENABLE_EXPERT_PARALLEL
MM_ENCODER_TP_MODE=$MM_ENCODER_TP_MODE
MTP_NUM_SPECULATIVE_TOKENS=$MTP_NUM_SPECULATIVE_TOKENS
MTP_DISABLE_BLOCK_DROP=$MTP_DISABLE_BLOCK_DROP
MTP_INDEX_SHARE=$MTP_INDEX_SHARE
MTP_DRAFT_VOCAB="$MTP_DRAFT_VOCAB"
IMAGE="$IMAGE"
PLE_OFFLOAD=false
QSA_PROFILE=$QSA_PROFILE
REQUIRE_IDLE_GPU=$REQUIRE_IDLE_GPU
EVICT_PAGE_CACHE=$EVICT_PAGE_CACHE
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
NFS_SHARE=false
EOF

cd "$RUNBOOK"
"$RUNBOOK/start.sh" --launch
rc=$?
[ $rc -ne 0 ] && exit "$rc"
# Their launcher exits once /health returns 200; keep the systemd unit
# alive (Type=simple) while the engine serves. The watchdog owns engine
# health; stop is handled by spark-vllm-podman-stop.
exec sleep infinity
