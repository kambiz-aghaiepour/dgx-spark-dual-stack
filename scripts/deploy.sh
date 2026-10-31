#!/bin/bash
# deploy.sh — install the dual-stack orchestration onto the head node.
#
# Usage:
#   ./deploy.sh                 # this node (head)
#   ./deploy.sh --all           # also sync runtime helpers to the worker
#                               # (runbook + image: see podman-adapt-runbook.sh)
#   ./deploy.sh --force-config  # replace the live model.conf with the template
#   ./deploy.sh --skip-mods     # don't install launcher mods into the docker fork
#
# Flags combine: ./deploy.sh --all --skip-mods.
#
# Runtime state is never clobbered silently: an existing
# ~/.config/spark-vllm/model.conf is kept unless --force-config is given, and
# the vllm-model-select launcher is always installed as the venv wrapper (never
# as a symlink to the .py, which breaks on `import urwid`).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_ALIAS="${HOST_ALIAS:-gx10-02}"
HEAD_IP="${HEAD_IP:-192.168.177.11}"
WORKER_IP="${WORKER_IP:-192.168.177.12}"
SPARK_VLLM_DOCKER="${SPARK_VLLM_DOCKER:-$HOME/spark-vllm-docker}"

ALL=0
FORCE_CONFIG=0
SKIP_MODS=0
for arg in "$@"; do
  case "$arg" in
    --all)          ALL=1 ;;
    --force-config) FORCE_CONFIG=1 ;;
    --skip-mods)    SKIP_MODS=1 ;;
    -h|--help)      sed -n '2,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)              echo "deploy.sh: unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

echo "==> installing scripts -> ~/.local/bin"
install -m 0755 "$HERE"/scripts/{spark-vllm-podman-start-select,spark-vllm-podman-start,spark-vllm-podman-stop,spark-vllm-podman-health,spark-vllm-watchdog,miaai_launch.sh,vllm-model-set,switch.sh} \
  "$HOME/.local/bin/"
install -m 0644 "$HERE"/scripts/{vllm-model-select.py,vllm-model-add.py} "$HOME/.local/bin/"
# the archive helper is also executed directly (over ssh from the head), so it
# needs the executable bit and its shebang, not just python importability
install -m 0755 "$HERE"/scripts/vllm-model-archive.py "$HOME/.local/bin/"
# and the name the docs / ssh one-liners use
install -m 0755 "$HERE"/scripts/vllm-model-archive "$HOME/.local/bin/"
install -m 0755 "$HERE"/scripts/spark-vllm-evict-cache.py "$HOME/.local/bin/"
# second stack: the switcher plus the TensorFold unit's start/stop/health/watchdog wrappers
install -m 0755 "$HERE"/scripts/{spark-stack-switch,spark-tensorfold-start,spark-tensorfold-stop,spark-tensorfold-health,spark-tensorfold-watchdog} \
  "$HOME/.local/bin/"
install -m 0755 "$HERE"/scripts/spark-vllm-warmup.sh "$HOME/"

# vllm-model-select is a wrapper around the .py (system python has no urwid).
# Remove a legacy symlink first: `install` would otherwise follow it and
# overwrite vllm-model-select.py itself.
rm -f "$HOME/.local/bin/vllm-model-select"
"$HERE"/scripts/setup-tui-venv.sh
install -m 0755 "$HERE"/scripts/vllm-model-select "$HOME/.local/bin/vllm-model-select"

echo "==> installing config -> ~/.config/spark-vllm"
mkdir -p "$HOME/.config/spark-vllm/flags"
if [ "$FORCE_CONFIG" = 1 ] || [ ! -e "$HOME/.config/spark-vllm/model.conf" ]; then
  install -m 0600 "$HERE"/config/model.conf.template "$HOME/.config/spark-vllm/model.conf"
  echo "    model.conf written from template"
else
  echo "    keeping existing model.conf (runtime state; --force-config replaces it)"
fi
install -m 0644 "$HERE"/config/flags/* "$HOME/.config/spark-vllm/flags/"
# which node this is, so a helper run locally (the worker over ssh) does not
# mistake itself for the head and fail to "reach" the worker
printf 'NODE=%s\n' "${SPARK_VLLM_NODE_ROLE:-head}" > "$HOME/.config/spark-vllm/node.conf"
chmod 0644 "$HOME/.config/spark-vllm/node.conf"

if [ "$SKIP_MODS" = 1 ]; then
  echo "==> skipping launcher mods (--skip-mods); start will fail unless they are present"
else
  echo "==> installing launcher mods -> $SPARK_VLLM_DOCKER/mods"
  if [ ! -d "$SPARK_VLLM_DOCKER" ]; then
    echo "    missing $SPARK_VLLM_DOCKER; clone kambiz-aghaiepour/spark-vllm-docker first" >&2
    echo "    (or re-run with --skip-mods for a scripts-only install)" >&2
    exit 1
  fi
  if [ ! -d "$SPARK_VLLM_DOCKER/mods/instanttensor-hybrid-draft-loader" ]; then
    echo "    missing $SPARK_VLLM_DOCKER/mods/instanttensor-hybrid-draft-loader" >&2
    echo "    update the spark-vllm-docker clone (it ships that mod)" >&2
    exit 1
  fi
  install -d "$SPARK_VLLM_DOCKER/mods"
  rm -rf "$SPARK_VLLM_DOCKER/mods/busy-loop-fix"
  cp -a "$HERE/scripts/mods/busy-loop-fix" "$SPARK_VLLM_DOCKER/mods/"
  echo "    busy-loop-fix installed (applied at launch by spark-vllm-podman-start)"
fi

echo "==> installing systemd user units (head)"
mkdir -p "$HOME/.config/systemd/user/spark-vllm-podman.service.d"
install -m 0644 "$HERE"/systemd/spark-vllm-podman.service "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/spark-vllm-health.service "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/spark-vllm-warmup.service "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/{spark-tensorfold.service,spark-tensorfold-watchdog.service} "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/{node-exporter,nvidia-gpu-exporter}.service "$HOME/.config/systemd/user/"
install -m 0644 "$HERE"/systemd/override.conf "$HOME/.config/systemd/user/spark-vllm-podman.service.d/"
systemctl --user daemon-reload
echo "    start-select dispatch wired via override.conf (head only)"
echo "    warmup service packaged; optional exporters: systemctl --user enable --now node-exporter nvidia-gpu-exporter"
# The TensorFold units ship disabled: they only run while stack.conf says tensorfold, and
# spark-stack-switch enables/starts them on the switch. Enabling them here would start the
# second stack at boot beside vLLM.
systemctl --user enable spark-tensorfold.service spark-tensorfold-watchdog.service >/dev/null 2>&1 || true
echo "    tensorfold units packaged and enabled (both no-op unless stack.conf says tensorfold)"
echo "    switch stacks with: spark-stack-switch tensorfold --apply   (back: spark-stack-switch vllm)"

if [ "$ALL" = 1 ]; then
  echo "==> syncing scripts to worker ($HOST_ALIAS)"
  ssh -o BatchMode=yes "$HOST_ALIAS" 'mkdir -p ~/.local/bin'
  scp -q "$HERE"/scripts/{spark-vllm-podman-stop,miaai_launch.sh,vllm-model-set} "$HOST_ALIAS":~/.local/bin/ 2>/dev/null || true
  # the archive tool is per node: each spark has its own disk and its own
  # ARCHIVE_PATH, so the worker gets the helper too (configure it there).
  scp -q "$HERE"/scripts/{vllm-model-archive.py,vllm-model-archive} "$HOST_ALIAS":~/.local/bin/ 2>/dev/null || true
  ssh -o BatchMode=yes "$HOST_ALIAS" 'chmod 755 ~/.local/bin/vllm-model-archive.py ~/.local/bin/vllm-model-archive' 2>/dev/null || true
  ssh -o BatchMode=yes "$HOST_ALIAS" 'mkdir -p ~/.config/spark-vllm && printf "NODE=worker\n" > ~/.config/spark-vllm/node.conf && chmod 0644 ~/.config/spark-vllm/node.conf' 2>/dev/null || true
  echo "    (worker only needs runtime helpers; full launch is driven from head)"
  echo "    (worker archive: ssh $HOST_ALIAS vllm-model-archive --set-path /archive)"
fi

echo "==> done. Verify with ./scripts/verify-deployment.sh then vllm-model-select --status"
