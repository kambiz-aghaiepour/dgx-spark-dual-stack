#!/bin/bash
# deploy.sh — install the dual-stack orchestration onto the head node.
# Usage:
#   ./deploy.sh          # this node (head)
#   ./deploy.sh --all    # also sync to worker (scripts + config only; runbook
#                        # and image are installed by podman-adapt-runbook.sh)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_ALIAS="${HOST_ALIAS:-gx10-02}"
HEAD_IP="${HEAD_IP:-192.168.177.11}"
WORKER_IP="${WORKER_IP:-192.168.177.12}"

echo "==> installing scripts -> ~/.local/bin"
install -m 0755 "$HERE"/scripts/{spark-vllm-podman-start-select,spark-vllm-podman-start,spark-vllm-podman-stop,spark-vllm-podman-health,spark-vllm-watchdog,miaai_launch.sh,vllm-model-set} \
  "$HOME/.local/bin/"
install -m 0644 "$HERE"/scripts/{vllm-model-select.py,vllm-model-add.py} "$HOME/.local/bin/"
install -m 0755 "$HERE"/scripts/spark-vllm-evict-cache.py "$HOME/.local/bin/"
ln -sf "$HOME/.local/bin/vllm-model-select.py" "$HOME/.local/bin/vllm-model-select"

echo "==> installing config -> ~/.config/spark-vllm"
mkdir -p "$HOME/.config/spark-vllm/flags"
install -m 0600 "$HERE"/config/model.conf.template "$HOME/.config/spark-vllm/model.conf"
install -m 0644 "$HERE"/config/flags/* "$HOME/.config/spark-vllm/flags/"

echo "==> installing systemd user units (head)"
mkdir -p "$HOME/.config/systemd/user/spark-vllm-podman.service.d"
install -m 0644 "$HERE"/systemd/spark-vllm-podman.service "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/spark-vllm-health.service "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/override.conf "$HOME/.config/systemd/user/spark-vllm-podman.service.d/"
systemctl --user daemon-reload
echo "    start-select dispatch wired via override.conf (head only)"

if [ "${1:-}" = "--all" ]; then
  echo "==> syncing scripts to worker ($HOST_ALIAS)"
  ssh -o BatchMode=yes "$HOST_ALIAS" 'mkdir -p ~/.local/bin'
  scp -q "$HERE"/scripts/{spark-vllm-podman-stop,miaai_launch.sh,vllm-model-set} "$HOST_ALIAS":~/.local/bin/ 2>/dev/null || true
  echo "    (worker only needs runtime helpers; full launch is driven from head)"
fi

echo "==> done. Next: vllm-model-set <model-id> --apply"
echo "    (model.conf currently bootstrapped from template — edit or set)"
