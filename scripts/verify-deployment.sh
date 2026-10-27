#!/bin/bash
# verify-deployment.sh — parity check between this repo tree and the live head.
#
# Compares every deployed artifact (repo copy vs installed copy), then probes the
# parts that are not plain files: the venv-backed selector launcher, urwid in
# that venv, the launcher mods in the docker fork, the systemd user units, and a
# live `vllm-model-select --status` run.
#
# Read-only: installs nothing, restarts nothing. Safe to run on a serving head.
#
# Usage: ./scripts/verify-deployment.sh
# Exit:  0 = parity, 1 = drift/missing pieces (each line says which)
set -uo pipefail   # deliberately not -e: run every check, then aggregate

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${HOME}/.local/bin"
CFG="${HOME}/.config/spark-vllm"
UNITS="${HOME}/.config/systemd/user"
DOCKER="${SPARK_VLLM_DOCKER:-$HOME/spark-vllm-docker}"
VENV="${HOME}/.local/share/vllm-tui/venv"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

fail=0
warn=0
pass=0

ok()   { pass=$((pass + 1)); printf 'OK   %s\n' "$*"; }
bad()  { fail=$((fail + 1)); printf 'FAIL %s\n' "$*"; }
note() { warn=$((warn + 1)); printf 'WARN %s\n' "$*"; }

check_file() {
  local repo="$1" live="$2"
  if [ ! -e "$live" ]; then
    bad "$live missing (repo has $repo)"
  elif cmp -s "$repo" "$live"; then
    ok "$live"
  else
    bad "$live differs from $repo"
  fi
}

echo "== repo: $HERE"
echo "== head: $(hostname) as $(id -un)"

echo
echo "-- deployed scripts"
for s in vllm-model-select.py vllm-model-set spark-vllm-watchdog \
         spark-vllm-podman-start-select spark-vllm-podman-start \
         spark-vllm-podman-stop spark-vllm-podman-health \
         spark-vllm-evict-cache.py vllm-model-add.py miaai_launch.sh; do
  check_file "$HERE/scripts/$s" "$BIN/$s"
done
check_file "$HERE/scripts/spark-vllm-warmup.sh" "$HOME/spark-vllm-warmup.sh"

echo
echo "-- per-model flags"
for f in "$HERE"/config/flags/*; do
  check_file "$f" "$CFG/flags/$(basename "$f")"
done

echo
echo "-- systemd user units"
for u in spark-vllm-podman.service spark-vllm-health.service spark-vllm-warmup.service; do
  check_file "$HERE/systemd/$u" "$UNITS/$u"
done
check_file "$HERE/systemd/override.conf" "$UNITS/spark-vllm-podman.service.d/override.conf"

echo
echo "-- selector launcher + venv"
# Content differs by design (the repo copy uses $HOME), so check the properties
# that matter instead of a byte comparison.
if [ -L "$BIN/vllm-model-select" ]; then
  bad "$BIN/vllm-model-select is a symlink (must be the venv wrapper; points at $(readlink "$BIN/vllm-model-select"))"
elif [ ! -x "$BIN/vllm-model-select" ]; then
  bad "$BIN/vllm-model-select missing or not executable"
elif grep -q 'vllm-tui/venv/bin/python' "$BIN/vllm-model-select" \
     && grep -q 'vllm-model-select\.py' "$BIN/vllm-model-select"; then
  ok "$BIN/vllm-model-select (venv wrapper)"
else
  bad "$BIN/vllm-model-select is not the venv wrapper"
fi

if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c 'import urwid' >/dev/null 2>&1; then
  ok "urwid $("$VENV/bin/python" -c 'import urwid; print(urwid.__version__)' 2>/dev/null) in $VENV"
else
  bad "urwid missing from $VENV (run scripts/setup-tui-venv.sh)"
fi

echo
echo "-- launcher mods ($DOCKER/mods)"
for m in busy-loop-fix instanttensor-hybrid-draft-loader; do
  if [ -d "$DOCKER/mods/$m" ]; then
    ok "$DOCKER/mods/$m"
  else
    bad "$DOCKER/mods/$m missing"
  fi
done
if [ -d "$DOCKER/mods/busy-loop-fix" ] && ! cmp -s \
     "$HERE/scripts/mods/busy-loop-fix/run.sh" "$DOCKER/mods/busy-loop-fix/run.sh"; then
  bad "$DOCKER/mods/busy-loop-fix/run.sh differs from $HERE/scripts/mods/busy-loop-fix/run.sh"
fi

echo
echo "-- systemd user units enabled"
for u in spark-vllm-podman.service spark-vllm-health.service spark-vllm-warmup.service; do
  if systemctl --user is-enabled "$u" >/dev/null 2>&1; then
    ok "$u enabled"
  else
    bad "$u not enabled"
  fi
done

echo
echo "-- runtime state (not repo artifacts)"
if [ -s "$CFG/model.conf" ]; then
  ok "$CFG/model.conf present ($(grep -m1 '^MODEL_ID=' "$CFG/model.conf" | cut -d= -f2-))"
else
  bad "$CFG/model.conf missing"
fi
if [ -e "$BIN/switch.sh" ]; then
  check_file "$HERE/scripts/switch.sh" "$BIN/switch.sh"
else
  note "$BIN/switch.sh not installed (optional convenience; deploy.sh installs it)"
fi

echo
echo "-- live status probe"
if status_out="$("$BIN/vllm-model-select" --status 2>&1)"; then
  ok "vllm-model-select --status"
else
  bad "vllm-model-select --status failed:"
  printf '%s\n' "$status_out" | sed 's/^/     /'
fi

echo
if [ "$fail" -gt 0 ]; then
  echo "PARITY: FAILED — $fail failed, $warn warning(s), $pass passed"
  exit 1
fi
echo "PARITY: OK — $pass checks passed, $warn warning(s)"
exit 0
