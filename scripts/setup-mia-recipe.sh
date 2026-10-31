#!/bin/bash
# Clone (or refresh) the pinned MiaAI-Lab DeepSeek-V4.1-Flash-1017 recipe and point it at
# this cluster. The recipe is third-party (Apache-2.0) and lives outside this repo.
#
#   scripts/setup-mia-recipe.sh          # clone/checkout the pin, install local.sh if absent
#   MIA_DIR=/path scripts/setup-mia-recipe.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
MIA_DIR="${SPARK_MIA_DIR:-$HOME/mia-dsv41}"
MIA_REPO="${MIA_REPO:-https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks.git}"
MIA_REF="${MIA_REF:-0e7d421}"
ENV_EXAMPLE="$ROOT/config/mia-dsv41.env.example"

if [ -d "$MIA_DIR/.git" ]; then
  echo "==> updating $MIA_DIR"
  git -C "$MIA_DIR" fetch -q origin
else
  echo "==> cloning $MIA_REPO -> $MIA_DIR"
  git clone -q "$MIA_REPO" "$MIA_DIR"
fi
git -C "$MIA_DIR" checkout -q "$MIA_REF"
git -C "$MIA_DIR" submodule update --init --recursive -q 2>/dev/null || true

if [ ! -f "$MIA_DIR/scripts/local.sh" ]; then
  install -m 0644 "$ENV_EXAMPLE" "$MIA_DIR/scripts/local.sh"
  echo "==> wrote $MIA_DIR/scripts/local.sh (edit WORKER / HF_CACHE / worker dirs)"
else
  echo "==> keeping existing $MIA_DIR/scripts/local.sh"
fi

echo
echo "Next: edit $MIA_DIR/scripts/local.sh, then run:"
echo "  cd $MIA_DIR && ./start.sh          # first run builds the image and downloads ~400 GB"
echo "  cd ~/dgx-spark-dual-stack && spark-stack-switch tensorfold --apply"
