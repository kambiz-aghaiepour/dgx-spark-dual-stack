#!/bin/bash
# setup-tui-venv.sh — create the interpreter venv used by vllm-model-select.
#
# The selector TUI imports urwid, which DGX OS system python does not ship.
# The venv is local tooling under ~/.local/share (not tracked by this repo);
# this script is the reproducible way to produce it.
#
# Usage: ./scripts/setup-tui-venv.sh [-f]
#   -f, --force   recreate the venv even if urwid is already importable
set -euo pipefail

VENV="${HOME}/.local/share/vllm-tui/venv"
URWID_VERSION="${URWID_VERSION:-4.2.5}"
FORCE=0

case "${1:-}" in
  -f|--force) FORCE=1 ;;
  "") ;;
  *)
    echo "usage: $0 [-f|--force]" >&2
    exit 2
    ;;
esac

if [ "$FORCE" = 0 ] && [ -x "$VENV/bin/python" ] \
   && "$VENV/bin/python" -c 'import urwid' >/dev/null 2>&1; then
  echo "==> urwid already available in $VENV"
  exit 0
fi

mkdir -p "$(dirname "$VENV")"
[ "$FORCE" = 1 ] && rm -rf "$VENV"

# Preferred path: stdlib venv + pip (works on a normal DGX OS install).
have_pip=0
if python3 -m venv "$VENV" >/dev/null 2>&1 \
   && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  have_pip=1
fi

if [ "$have_pip" = 1 ]; then
  "$VENV/bin/python" -m pip install --upgrade pip >/dev/null
  "$VENV/bin/pip" install "urwid==${URWID_VERSION}"
elif command -v uv >/dev/null 2>&1; then
  # Images without ensurepip: fall back to uv (already present on the Spark).
  echo "==> python3 -m venv/pip unavailable here; using uv"
  rm -rf "$VENV"
  uv venv "$VENV"
  uv pip install --python "$VENV/bin/python" "urwid==${URWID_VERSION}"
else
  echo "setup-tui-venv: cannot create $VENV" >&2
  echo "  python3 -m venv/pip failed and uv is not on PATH" >&2
  exit 1
fi

"$VENV/bin/python" -c 'import urwid, sys; print("==> urwid %s on python %s in %s" % (urwid.__version__, sys.version.split()[0], sys.prefix))'
