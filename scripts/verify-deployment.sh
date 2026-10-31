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
         spark-vllm-evict-cache.py vllm-model-add.py miaai_launch.sh \
         spark-stack-switch spark-tensorfold-start spark-tensorfold-stop \
         spark-tensorfold-health spark-tensorfold-watchdog; do
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
for u in spark-vllm-podman.service spark-vllm-health.service spark-vllm-warmup.service \
         spark-tensorfold.service spark-tensorfold-watchdog.service; do
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
for u in spark-vllm-podman.service spark-vllm-health.service spark-vllm-warmup.service \
         spark-tensorfold.service spark-tensorfold-watchdog.service; do
  if systemctl --user is-enabled "$u" >/dev/null 2>&1; then
    ok "$u enabled"
  else
    bad "$u not enabled"
  fi
done

echo
echo "-- stack assignment (vllm or tensorfold)"
if [ -f "$CFG/stack.conf" ]; then
  ok "stack.conf: STACK=$(sed -n 's/^STACK=//p' "$CFG/stack.conf" | head -1) TF_PROFILE=$(sed -n 's/^TF_PROFILE=//p' "$CFG/stack.conf" | head -1)"
else
  note "no $CFG/stack.conf (means vllm); deploy.sh ships the switcher, spark-stack-switch writes it"
fi
if [ -x "$BIN/spark-stack-switch" ]; then
  ok "spark-stack-switch present"
else
  bad "spark-stack-switch missing"
fi

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
echo "-- archive tool"
if grep -qx 'NODE=head' "$HOME/.config/spark-vllm/node.conf" 2>/dev/null; then
  ok "node.conf says NODE=head"
else
  note "node.conf missing or not NODE=head (deploy.sh writes it; the archive helper needs it to know which node it is)"
fi

if [ -e "$BIN/vllm-model-archive.py" ]; then
  check_file "$HERE/scripts/vllm-model-archive.py" "$BIN/vllm-model-archive.py"
  check_file "$HERE/scripts/vllm-model-archive" "$BIN/vllm-model-archive"
  if arc_json="$(python3 "$BIN/vllm-model-archive.py" --status --json 2>&1)"; then

    arc_lines="$(printf '%s' "$arc_json" | python3 -c '
import json, sys


def human(n):
    n = float(n or 0)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or u == "TB":
            return ("%.1f %s" % (n, u)) if u != "B" else "%d B" % int(n)
        n /= 1000.0
    return "%.1f TB" % n


try:
    d = json.load(sys.stdin)
except Exception:
    print("  (unparsable status)")
    raise SystemExit
for n, i in (d.get("nodes") or {}).items():
    if not i.get("ok", True):
        print("  %-7s %s" % (n, i.get("error", "unavailable")))
        continue
    if not i.get("configured"):
        print("  %-7s not configured" % n)
        continue
    state = "mounted" if i.get("mounted") else "NOT mounted"
    print("  %-7s %s | %s | %d archived | cache free %s"
          % (n, i.get("archive_path"), state, len(i.get("archived") or []),
             human(i.get("cache_free_bytes"))))
' 2>/dev/null)"
    ok "archive status:"
    printf '%s\n' "${arc_lines:-  ?}" | while IFS= read -r l; do printf '    %s\n' "$l"; done
  else
    bad "vllm-model-archive --status failed"
  fi
else
  note "$BIN/vllm-model-archive.py not installed (deploy.sh installs it; archive/unarchive disabled)"
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
