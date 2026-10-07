#!/bin/bash
# switch.sh — thin wrapper around vllm-model-set / vllm-model-select so the
# daily "which stack am I on / switch me" flow is one obvious command.
# Usage:
#   switch.sh status
#   switch.sh list
#   switch.sh to <model-id> [alias] [--apply]
set -euo pipefail
BIN="$HOME/.local/bin"
ALIAS="${HOST_ALIAS:-gx10-01}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

cmd="${1:-status}"; shift || true
case "$cmd" in
  status) exec "$BIN/vllm-model-select" --status ;;
  list)   exec "$BIN/vllm-model-select" --list ;;
  to)
    id="${1:?usage: switch.sh to <model-id> [alias] [--apply]}"; shift || true
    exec "$BIN/vllm-model-set" "$id" "$@" ;;
  *)
    echo "usage: switch.sh {status|list|to <model-id> [alias] [--apply]}" >&2
    exit 2 ;;
esac
