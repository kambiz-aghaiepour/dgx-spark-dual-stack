#!/bin/bash
# Clone (or refresh) the pinned GLM-5.3-Flash recipe and point it at this cluster. The recipe is third-party and lives
# outside this repo, next to ~/mia-dsv41. The pin is our fork of MiaAI-Lab's recipe: it adds the container-runtime
# switch the pair runs on (CONTAINER_RT=podman; upstream PR MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold#246).
#
# GLM is the second TensorFold profile (TF_PROFILE=glm53): both profiles run under one systemd unit,
# spark-tensorfold.service, and spark-stack-switch decides which one is live and what starts at boot.
#
#   scripts/setup-glm-recipe.sh            # clone/checkout the pin, write local.sh + the gated token file if absent
#   GLM_DIR=/path scripts/setup-glm-recipe.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
GLM_DIR="${SPARK_GLM_DIR:-$HOME/glm53}"
GLM_REPO="${GLM_REPO:-https://github.com/kambiz-aghaiepour/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold.git}"
# podman-support: ebdce32 (runtime switch, rootful store reads, token by value, kernel-cache mount source; PR #246)
GLM_REF="${GLM_REF:-ebdce32}"
ENV_EXAMPLE="$ROOT/config/glm53.env.example"

if [ -d "$GLM_DIR/.git" ]; then
  have=$(git -C "$GLM_DIR" remote get-url origin 2>/dev/null || true)
  if [ "$have" != "$GLM_REPO" ]; then
    echo "==> origin: $have -> $GLM_REPO"
    git -C "$GLM_DIR" remote set-url origin "$GLM_REPO"
  fi
  echo "==> updating $GLM_DIR"
  git -C "$GLM_DIR" fetch -q origin
else
  echo "==> cloning $GLM_REPO -> $GLM_DIR"
  git clone -q "$GLM_REPO" "$GLM_DIR"
fi
git -C "$GLM_DIR" checkout -q "$GLM_REF"

if [ ! -f "$GLM_DIR/scripts/local.sh" ]; then
  install -m 0644 "$ENV_EXAMPLE" "$GLM_DIR/scripts/local.sh"
  echo "==> wrote $GLM_DIR/scripts/local.sh (edit WORKER / HF_CACHE / worker dirs)"
else
  echo "==> keeping existing $GLM_DIR/scripts/local.sh"
fi

# The Ablit weights are gated, and start.sh insists on a token even when they are already on disk. Keep it out of the
# checkout (which is a git clone) and out of the unit file: the start wrapper loads this 0600 file for the glm53 profile.
ENVF="${SPARK_GLM_ENV:-$HOME/.config/spark/glm53.env}"
if [ ! -f "$ENVF" ]; then
  if tok=$(cat "$HOME/.cache/huggingface/token" 2>/dev/null) && [ -n "$tok" ]; then
    mkdir -p "$(dirname "$ENVF")"
    (umask 077 && printf 'HF_TOKEN=%s\n' "$tok" > "$ENVF")
    echo "==> wrote $ENVF (0600, from ~/.cache/huggingface/token)"
  else
    echo "==> no ~/.cache/huggingface/token: write HF_TOKEN=hf_... into $ENVF before the first start" >&2
  fi
else
  echo "==> keeping existing $ENVF"
fi

echo
echo "Next: edit $GLM_DIR/scripts/local.sh (WORKER / HF_CACHE / CONTAINER_RT), then run:"
echo "  cd $GLM_DIR && ./scripts/prepare.sh     # pulls the pinned image and the ~180 GB gated checkpoint"
echo "  cd ~/dgx-spark-dual-stack && spark-stack-switch tensorfold glm53 --apply"
