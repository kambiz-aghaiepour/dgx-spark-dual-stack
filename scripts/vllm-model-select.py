#!/usr/bin/env python3
"""vllm-model-select - Phase 2 TUI (urwid) for the DGX cluster.

Lists models known to the cluster (cached, archived, or both), picks the boot
default and a served-model-name alias, writes model.conf, or applies via
vllm-model-set.  Archived models stay listed but greyed out: they cannot be
selected until they are unarchived.

Non-interactive:
  vllm-model-select --list
  vllm-model-select --status
  vllm-model-select --archive <model-id> | --unarchive <model-id>
  vllm-model-select --set-archive-path /archive
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import urwid

CONF = os.path.expanduser("~/.config/spark-vllm/model.conf")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
WORKER = "gx10-02"
PATH_SETTER = os.path.expanduser("~/.local/bin/vllm-model-set")
LOCKS = os.path.expanduser("~/.config/spark-vllm/locked.conf")
FLAGS_DIR = os.path.expanduser("~/.config/spark-vllm/flags")
ARCHIVER = os.path.expanduser("~/.local/bin/vllm-model-archive.py")
ARCHIVE_LOG = os.path.expanduser("~/.config/spark-vllm/archive.log")


def human(n):
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000.0
    return f"{n:.1f} TB"


def archive_call(*args, timeout=30):
    """Run vllm-model-archive with --json and return its parsed result."""
    if not os.path.exists(ARCHIVER):
        return {"ok": False, "error": "vllm-model-archive.py not installed (run deploy.sh)",
                "models": [], "configured": False, "archive_path": ""}
    try:
        r = subprocess.run([sys.executable, ARCHIVER, *args, "--json"],
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "archive helper timed out"}
    try:
        return json.loads(r.stdout)
    except ValueError:
        return {"ok": False, "error": (r.stderr or r.stdout).strip()[:200]}


def tree_bytes(path):
    """Apparent size of a cache tree, hardlinks once, symlinks not followed."""
    total, seen = 0, set()
    for root, dirs, names in os.walk(path, followlinks=False):
        for n in names + dirs:
            p = os.path.join(root, n)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
    return total


def archive_list():
    res = archive_call("--list", timeout=90)
    if res.get("models"):
        return res
    # helper missing: fall back to a plain cache scan so the TUI still works
    conf = archive_call("--status")
    rows = []
    for m in scan_models():
        rd = os.path.join(HUB, "models--" + m["id"].replace("/", "--"))
        rows.append({"id": m["id"],
                     "engine": "miaai" if "Qwen3.8-Flash-Next-NVFP4" in m["id"] else "b12x",
                     "state": "cached", "cached": True, "archived": False,
                     "cache_bytes": tree_bytes(rd), "archive_bytes": 0,
                     "current": False})
    try:
        free = shutil.disk_usage(HUB).free
    except OSError:
        free = 0
    return {"ok": False, "configured": conf.get("configured", False),
            "archive_path": conf.get("archive_path", ""),
            "cache_free_bytes": free, "models": rows,
            "note": "archive tool unavailable - run deploy.sh to enable x/u/p"}


def safe_name(mid):
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", mid.replace("/", "-"))[:60]


def locks():
    try:
        return set(l.strip() for l in open(LOCKS)
                   if l.strip() and not l.startswith("#"))
    except FileNotFoundError:
        return set()


def set_lock(mid, on):
    s = locks()
    (s.add if on else s.discard)(mid)
    with open(LOCKS, "w") as f:
        f.write("\n".join(sorted(s)) + ("\n" if s else ""))


def gc_orphan_blobs():
    """Remove hub/blobs entries no snapshot symlink references (frees space)."""
    refs = set()
    for p in glob.glob(os.path.join(HUB, "models--*", "snapshots", "*")):
        try:
            for name in os.listdir(p):
                lp = os.path.join(p, name)
                if os.path.islink(lp):
                    refs.add(os.path.realpath(lp))
        except OSError:
            pass
    freed = 0
    for p in glob.glob(os.path.join(HUB, "blobs", "*")):
        if os.path.isfile(p) and p not in refs:
            try:
                freed += os.path.getsize(p)
                os.unlink(p)
            except OSError:
                pass
    return freed


def delete_model(mid, confirm=True):
    cur = read_conf().get("MODEL_ID", "")
    if mid == cur:
        return f"REFUSED: {mid} is the currently used model - switch to another first"
    if mid in locks():
        return f"REFUSED: {mid} is locked - unlock first (vllm-model-select --unlock {mid})"
    if confirm:
        ans = input(f"DELETE {mid} and all its cached data? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            return "canceled"
    hdir = "models--" + mid.replace("/", "--")
    subprocess.run(["rm", "-rf", os.path.join(HUB, hdir)])
    subprocess.run(["ssh", "-o", "BatchMode=yes", WORKER,
                    f"rm -rf ~/.cache/huggingface/hub/{hdir}"], capture_output=True)
    flags = os.path.join(FLAGS_DIR, safe_name(mid) + ".conf")
    if os.path.exists(flags):
        os.unlink(flags)
    set_lock(mid, False)
    freed = gc_orphan_blobs()
    return (f"deleted {mid} (cache + worker copy + flags + "
            f"unreferenced blobs, freed ~{freed / 1e9:.1f} GB)")


def read_conf():
    d = {}
    try:
        for line in open(CONF):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k] = v
    except FileNotFoundError:
        pass
    return d


def scan_models():
    """Cached repos: id, worker size ('1.2 GB' or '')."""
    models = []
    for d in sorted(glob.glob(HUB + "/models--*")):
        name = os.path.basename(d)[len("models--"):]
        if "--" not in name:
            continue
        org, mod = name.split("--", 1)
        mid = f"{org}/{mod}"
        wmb = ""
        try:
            out = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4",
                 WORKER, f"du -sm {d} 2>/dev/null | cut -f1"],
                capture_output=True, text=True, timeout=12,
            ).stdout.strip()
            if out:
                wmb = f"{int(out) / 1024:.1f} GB"
        except Exception:
            pass
        models.append({"id": mid, "worker": wmb})
    return models


def run_list():
    res = archive_list()
    conf = read_conf()
    cur = conf.get("MODEL_ID", "")
    print(f"{'MODEL':58} {'ENGINE':6} {'STATE':16} {'CACHED':>9} {'ARCHIVE':>9}  status")
    for m in res["models"]:
        flags = []
        if m["id"] == cur:
            flags.append("current")
        if m["id"] in locks():
            flags.append("LOCKED")
        if not m.get("cached"):
            flags.append("archived - unarchive to use")
        print(f"{m['id']:<58} {m['engine']:<6} {m['state']:<16} "
              f"{human(m.get('cache_bytes')):>9} {human(m.get('archive_bytes')):>9}  {' '.join(flags)}")
    print(f"\ncurrent: {cur} -> {conf.get('SERVED_MODEL_NAME', '')}")
    arch = res.get("archive_path") or "(not configured - run --set-archive-path)"
    free = res.get("archive_free_bytes")
    print(f"archive: {arch}" + (f"  free {human(free)}" if free else ""))
    print(f"cache:   free {human(res.get('cache_free_bytes'))}")
    if res.get("note"):
        print(f"note:    {res['note']}")


class PickRow(urwid.Text):
    """Model row: selectable (arrows/j/k) and click-to-pick with the mouse."""

    def __init__(self, text, owner, mid):
        super().__init__(text)
        self._owner, self._mid = owner, mid

    def selectable(self):
        return True

    def keypress(self, size, key):
        return key

    def mouse_event(self, size, event, button, col, row, focus):
        if button == 1 and event in ("mouse press", "mouse drag", "mouse release"):
            self._owner.pick(self._mid)
            return True
        return False


class Selector:
    palette = [
        ("body", "light gray", "black"),
        ("head", "white,bold", "dark blue"),
        ("cur", "black,bold", "light cyan"),
        ("err", "light red", "black"),
        ("sel", "white,bold", "dark magenta"),
        ("arch", "dark gray", "black"),
        ("archsel", "light gray", "dark gray"),
    ]

    def __init__(self):
        self.conf = read_conf()
        self.arch = archive_list()
        self.models = self.arch.get("models", [])
        self.pending = self.conf.get("MODEL_ID", "")
        self.alias = self.conf.get("SERVED_MODEL_NAME", "")
        self.msg = ""
        self.editing = False
        self.editing_path = False
        self.picked = ""
        self.capture = True
        self.confirming = False
        self.confirming_del = False
        self.confirming_arch = False
        self.del_target = ""
        self.arch_target = ""
        self.bg = None
        self.bg_label = ""
        self.bg_out = None
        self.exec_apply = None
        self.body = urwid.ListBox(urwid.SimpleListWalker([]))
        self.footer = urwid.Text("", wrap="clip")
        self.refresh()

    def row_for(self, mid):
        for m in self.models:
            if m["id"] == mid:
                return m
        return {}

    def focus_row(self):
        if not self.models:
            return {}
        return self.row_for(self.models[min(self.body.focus_position, len(self.models) - 1)]["id"])

    def pick(self, mid):
        if not self.row_for(mid).get("cached"):
            self.msg = f"{mid} is archived (greyed out) - press u to unarchive it first"
            self.refresh()
            return
        self.pending = mid
        self.picked = mid
        self.msg = f"picked: {mid}  (a -> use as alias)"
        self.refresh()

    def build_rows(self):
        cur, rows = self.conf.get("MODEL_ID", ""), []
        for m in self.models:
            mark = ">" if m["id"] == self.pending else " "
            curm = "*" if m["id"] == cur else " "
            locked = " [LOCKED]" if m["id"] in locks() else ""
            archived = not m.get("cached")
            if archived:
                state, size = "ARCHIVED", human(m.get("archive_bytes"))
                hint = ("  fits" if m.get("fits") else "  NO-SPACE") if m.get("archived") else ""
            else:
                state, size = "cached", human(m.get("cache_bytes"))
                hint = ""
            text = (f"{curm}{mark} {m['id']:<46} {m['engine']:<5} {state:<8} {size:>9}"
                    f"{'  CURR' if m['id'] == cur else ''}{locked}{hint}")
            attr = "arch" if archived else ("cur" if m["id"] == self.pending else "body")
            rows.append(urwid.AttrMap(PickRow(text, self, m["id"]), attr,
                                      "archsel" if archived else "sel"))
        return rows

    def refresh(self):
        self.body.body[:] = self.build_rows()
        self.footer.set_text(
            f"{self.msg}  ||  alias: {self.alias}  click/Enter:pick  a:alias  "
            f"s:default  A:apply  x:archive  u:unarchive  p:archive-path  "
            f"d:delete  l:lock  m:mouse  r:refresh  q:quit")

    def reload(self):
        self.conf = read_conf()
        self.arch = archive_list()
        self.models = self.arch.get("models", [])
        if self.pending and not self.row_for(self.pending).get("cached"):
            self.pending = ""
        self.refresh()

    def switch_status(self):
        try:
            st = subprocess.run(
                ["systemctl", "--user", "show", "-p", "ActiveState", "--value",
                 "spark-vllm-podman.service"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            procs = subprocess.run(
                ["pgrep", "-af", "[v]llm-model-set"],
                capture_output=True, text=True, timeout=5).stdout
            running = [l for l in procs.splitlines()
                       if "--apply" in l and "bash" not in l
                       and "sh " not in l and "pgrep" not in l]
        except Exception:
            return ""
        if st == "activating" or running:
            return (f"\u26a0 SWITCH IN PROGRESS -> "
                    f"{self.conf.get('MODEL_ID', '?')} (service restarting)")
        if st == "active":
            return f"serving: {self.conf.get('MODEL_ID', '?')} (live)"
        if st == "failed":
            return "\u26a0 FAILED: service failed - check: journalctl --user -u spark-vllm-podman.service"
        return st

    def header_text(self):
        n_cache = sum(1 for m in self.models if m.get("cached"))
        n_arch = sum(1 for m in self.models if m.get("archived"))
        apath = self.arch.get("archive_path") or "(archive path not set - p to configure)"
        afree = self.arch.get("archive_free_bytes")
        lines = [f" vllm-model-select | current: {self.conf.get('MODEL_ID', '')} as "
                 f"{self.conf.get('SERVED_MODEL_NAME', '')} | {n_cache} cached, {n_arch} archived",
                 f" archive: {apath}" + (f" ({human(afree)} free)" if afree else "")
                 + f" | cache free {human(self.arch.get('cache_free_bytes'))}"]
        if self.bg:
            lines.append(f" \u23f3 {self.bg_label} running - footer shows the result")
        st = self.switch_status()
        if st:
            lines.append(f" {st}")
        return "\n".join(lines)

    def header(self):
        if not getattr(self, "head", None):
            self.head = urwid.Text(self.header_text())
        else:
            self.head.set_text(self.header_text())
        return urwid.AttrMap(self.head, "head")

    def apply_now(self):
        self.exec_apply = [PATH_SETTER, self.pending,
                           self.alias or self.pending, "--apply"]
        raise urwid.ExitMainLoop()

    def start_archive(self):
        if self.bg:
            self.msg = f"{self.bg_label} is still running"
            self.refresh()
            return True
        row = self.focus_row()
        if not row:
            return True
        mid = row["id"]
        if not self.arch.get("configured"):
            self.msg = "archive path not configured - press p to set it first"
            self.refresh()
            return True
        if not row.get("cached"):
            self.msg = f"{mid} is already archived"
            self.refresh()
            return True
        if row.get("current"):
            self.msg = "REFUSED: that is the current boot model - switch first"
            self.refresh()
            return True
        if mid in locks():
            self.msg = "REFUSED: model is locked (toggle with l)"
            self.refresh()
            return True
        self.arch_target, self.arch_action = mid, "archive"
        self.confirming_arch = True
        self.loop.screen.set_mouse_tracking(False)
        self.loop.widget = urwid.Overlay(
            urwid.LineBox(urwid.Filler(urwid.Text(
                f"ARCHIVE {mid}?\n"
                f"  engine: {row.get('engine')}\n"
                f"  move:   {human(row.get('cache_bytes'))} -> {self.arch.get('archive_path')}\n"
                f"  then the internal cache copy is removed.\n"
                f"  profiles (flags) stay; the row stays, greyed out.\n"
                f"Takes minutes for a full checkpoint. [y/N]", align="center")),
                title="confirm archive"),
            self.frame, "center", ("relative", 72), "middle", 8)
        return True

    def start_unarchive(self):
        if self.bg:
            self.msg = f"{self.bg_label} is still running"
            self.refresh()
            return True
        row = self.focus_row()
        if not row:
            return True
        mid = row["id"]
        if not row.get("archived"):
            self.msg = f"{mid} is not archived - nothing to pull back"
            self.refresh()
            return True
        if row.get("cached"):
            self.msg = f"{mid} is already in the cache"
            self.refresh()
            return True
        need = human(row.get("needed_bytes"))
        if row.get("fits") is False:
            sugg = ", ".join(s["id"] for s in row.get("suggestions") or []) or "another cached model"
            self.msg = (f"{mid} does not fit: needs {need}, usable free "
                        f"{human(row.get('fits_have_bytes'))} - archive {sugg} first")
            self.loop.widget = urwid.Overlay(
                urwid.LineBox(urwid.Filler(urwid.Text(
                    f"NOT ENOUGH INTERNAL SPACE for {mid}\n\n"
                    f"  needs:        {need}\n"
                    f"  usable free:  {human(row.get('fits_have_bytes'))}\n"
                    f"  short by:     {human(row.get('shortfall_bytes'))}\n\n"
                    f"Archive another model first (x), e.g.:\n  {sugg}\n\n"
                    f"[any key]", align="center")), title="unarchive blocked"),
                self.frame, "center", ("relative", 72), "middle", 10)
            self.refresh()
            return True
        self.arch_target, self.arch_action = mid, "unarchive"
        self.confirming_arch = True
        self.loop.screen.set_mouse_tracking(False)
        self.loop.widget = urwid.Overlay(
            urwid.LineBox(urwid.Filler(urwid.Text(
                f"UNARCHIVE {mid}?\n"
                f"  engine: {row.get('engine')}\n"
                f"  pull back: {need} into the internal cache\n"
                f"  free after: ~{human(max(0, (row.get('fits_have_bytes') or 0) - (row.get('needed_bytes') or 0)))}\n"
                f"[y/N]", align="center")), title="confirm unarchive"),
            self.frame, "center", ("relative", 72), "middle", 8)
        return True

    def run_bg(self, action, mid):
        try:
            os.makedirs(os.path.dirname(ARCHIVE_LOG), exist_ok=True)
            self.bg_out = open(ARCHIVE_LOG, "w")
        except OSError as e:
            self.msg = f"cannot write {ARCHIVE_LOG}: {e}"
            self.refresh()
            return
        self.bg = subprocess.Popen(
            [sys.executable, ARCHIVER, "--" + action, mid, "--json"],
            stdout=self.bg_out, stderr=subprocess.STDOUT)
        self.bg_label = f"{action} {mid}"
        self.msg = f"{self.bg_label} started (log {ARCHIVE_LOG})"
        self.refresh()

    def poll_bg(self):
        if not self.bg or self.bg.poll() is None:
            return
        label = self.bg_label
        try:
            if self.bg_out:
                self.bg_out.close()
        except OSError:
            pass
        try:
            with open(ARCHIVE_LOG) as f:
                res = json.load(f)
        except (OSError, ValueError):
            res = {}
        self.bg, self.bg_label, self.bg_out = None, "", None
        self.reload()
        if res.get("ok"):
            if "freed_bytes" in res:
                self.msg = (f"archived {res.get('model_id')}: {human(res.get('bytes'))} moved, "
                            f"freed {human(res.get('freed_bytes'))} internal")
            else:
                self.msg = (f"unarchived {res.get('model_id')}: "
                            f"{human(res.get('restored_bytes'))} pulled back")
        else:
            self.msg = f"{label} failed: {res.get('error') or 'see ' + ARCHIVE_LOG}"
        self.refresh()

    def edit_archive_path(self):
        self.editing_path = True
        self.loop.screen.set_mouse_tracking(False)
        edit = urwid.Edit("archive path: ", self.arch.get("archive_path") or "/archive")
        self._path_edit = edit
        self.loop.widget = urwid.Overlay(
            urwid.LineBox(urwid.Filler(edit),
                          title="archive path - enter=save, esc=cancel"),
            self.frame, "center", ("relative", 70), "middle", 3)
        return True

    def keypress(self, key):
        if self.confirming_arch:
            self.confirming_arch = False
            self.loop.widget = self.frame
            self.loop.screen.set_mouse_tracking(self.capture)
            if key in ("y", "Y"):
                self.run_bg(self.arch_action, self.arch_target)
            else:
                self.msg = f"{self.arch_action} canceled"
                self.refresh()
            return True
        if self.editing_path:
            if key == "enter":
                self.editing_path = False
                self.loop.widget = self.frame
                self.loop.screen.set_mouse_tracking(self.capture)
                path = self._path_edit.get_edit_text().strip()
                res = archive_call("--set-path", path)
                if res.get("ok"):
                    self.msg = f"archive path: {res['path']} ({human(res.get('free_bytes'))} free)"
                else:
                    self.msg = f"cannot use {path}: {res.get('error')}"
                self.reload()
            elif key == "esc":
                self.editing_path = False
                self.loop.widget = self.frame
                self.loop.screen.set_mouse_tracking(self.capture)
                self.msg = "archive path unchanged"
                self.refresh()
            return True
        if self.confirming_del:
            self.confirming_del = False
            self.loop.widget = self.frame
            if key in ("y", "Y"):
                self.msg = delete_model(self.del_target, confirm=False)
            else:
                self.msg = "delete canceled"
            self.refresh()
            return True
        if self.confirming:
            self.confirming = False
            self.loop.widget = self.frame
            if key in ("y", "Y"):
                return self.apply_now()
            self.msg = "apply cancelled"
            self.refresh()
            return True
        if self.editing:
            if key == "enter":
                self.editing = False
                self.loop.widget = self.frame
                self.loop.screen.set_mouse_tracking(True)
                self.msg = f"alias set: {self.alias}"
                self.refresh()
            elif key == "esc":
                self.editing = False
                self.loop.widget = self.frame
                self.loop.screen.set_mouse_tracking(True)
                self.msg = "alias unchanged"
                self.refresh()
            elif key == "ctrl u":
                self._edit.set_edit_text("")
                self._edit.set_edit_pos(0)
                self.msg = "alias cleared (serve under model id)"
                self.refresh()
            return True
        if key in ("q", "Q"):
            raise urwid.ExitMainLoop()
        if key in ("m", "M"):
            self.capture = not self.capture
            self.loop.screen.set_mouse_tracking(self.capture)
            self.msg = ("app mouse: click-to-pick" if self.capture
                        else "terminal mouse: drag-select + ctrl-shift-c")
            self.refresh()
            return True
        if key in ("r", "R"):
            self.conf = read_conf()
            self.models = scan_models()
            self.msg = "refreshed"
            self.refresh()
            return True
        if key == "a":
            self.editing = True
            self.loop.screen.set_mouse_tracking(False)
            edit = urwid.Edit("served-model-name (alias): ",
                              self.picked or self.alias)
            self._edit = edit

            def changed(widget, new):
                self.alias = new.strip() or self.alias

            urwid.connect_signal(edit, "change", changed)
            self.loop.widget = urwid.Overlay(
                urwid.LineBox(urwid.Filler(edit),
                              title="alias (enter=commit, ctrl-u=clear, esc=cancel)"),
                self.frame, "center", ("relative", 70), "middle", 5)
            return True
        if key in ("s", "S", "A"):
            if not self.pending:
                self.msg = "select a model first (Enter)"
                self.refresh()
                return True
            if not self.row_for(self.pending).get("cached"):
                self.msg = f"{self.pending} is archived - unarchive it (u) before using it"
                self.refresh()
                return True
            if key in ("A",):
                self.confirming = True
                self.loop.widget = urwid.Overlay(
                    urwid.LineBox(urwid.Filler(urwid.Text(
                        f"APPLY now?\n  model: {self.pending}\n"
                        f"  alias: {self.alias or '(none - own id)'}\n"
                        f"~5-10 min restart. [y/N]", align="center")),
                        title="confirm apply"),
                    self.frame, "center", ("relative", 70), "middle", 5)
                return True
            act = subprocess.run(
                [PATH_SETTER, self.pending, self.alias or self.pending],
                capture_output=True, text=True, timeout=60)
            self.conf = read_conf()
            if act.returncode != 0:
                self.msg = f"set failed: {(act.stderr or act.stdout).strip()[:90]}"
                self.refresh()
                return True
            self.msg = f"default written for {self.pending} (next boot)"
            self.refresh()
            return True
        if key == "enter":
            row = self.focus_row()
            if row:
                if not row.get("cached"):
                    self.msg = (f"{row['id']} is archived (greyed) - press u to unarchive it first")
                else:
                    self.pending = row["id"]
                    self.msg = f"pending: {self.pending}"
                self.refresh()
            return True
        if key in ("x", "X"):
            return self.start_archive()
        if key in ("u", "U"):
            return self.start_unarchive()
        if key in ("p", "P"):
            return self.edit_archive_path()
        if key in ("l", "L"):
            if self.models:
                mid = self.models[self.body.focus_position]["id"]
                was = mid in locks()
                set_lock(mid, not was)
                self.msg = f"{'locked' if not was else 'unlocked'} {mid}"
                self.refresh()
            return True
        if key in ("d", "D"):
            if self.models:
                mid = self.models[self.body.focus_position]["id"]
                cur = self.conf.get("MODEL_ID", "")
                if mid == cur:
                    self.msg = f"REFUSED: {mid} is the currently used model"
                elif not self.row_for(mid).get("cached"):
                    self.msg = (f"REFUSED: {mid} is archived - keep it, or delete the "
                                f"archive dir under the archive path")
                elif mid in locks():
                    self.msg = f"REFUSED: {mid} is locked (toggle with l)"
                else:
                    self.del_target, self.confirming_del = mid, True
                    self.loop.widget = urwid.Overlay(
                        urwid.LineBox(urwid.Filler(urwid.Text(
                            f"DELETE {mid} and ALL cached data (cache, flags,\n"
                            f"worker copy)? This cannot be undone. [y/N]",
                            align="center")), title="confirm delete"),
                        self.frame, "center", ("relative", 70), "middle", 5)
                self.refresh()
            return True
        if key in ("down", "j", "J") and self.body.focus_position < len(self.models) - 1:
            self.body.set_focus(self.body.focus_position + 1)
            return True
        if key in ("up", "k", "K") and self.body.focus_position > 0:
            self.body.set_focus(self.body.focus_position - 1)
            return True

    def _tick(self, loop, user_data):
        self.poll_bg()
        self.header()
        loop.set_alarm_in(3 if self.bg else 10, self._tick)

    def run(self):
        self.refresh()
        self.frame = urwid.Frame(urwid.AttrMap(self.body, "body"),
                                 header=self.header(), footer=self.footer)
        self.loop = urwid.MainLoop(self.frame, self.palette,
                                   unhandled_input=self.keypress)
        self.loop.set_alarm_in(10, self._tick)
        try:
            self.loop.run()
        except KeyboardInterrupt:
            self.loop.screen.stop()
            print("\n^C - exiting (selection unchanged)")
            sys.exit(130)
        self.loop.screen.stop()
        if self.exec_apply:
            print(f"\n>>> handing off: {' '.join(self.exec_apply)}\n")
            os.execv(self.exec_apply[0], self.exec_apply)


def run_status():
    import json
    import urllib.request

    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=6).stdout.strip()
        except Exception:
            return "?"

    conf = read_conf()
    unit = sh("systemctl", "--user", "show", "-p", "ActiveState", "--value",
              "spark-vllm-podman.service")
    wd = sh("systemctl", "--user", "is-active", "spark-vllm-health.service")
    procs = sh("pgrep", "-af", "[v]llm-model-set")
    applying = bool(procs and "--apply" in procs
                    and "bash" not in procs and "pgrep" not in procs)
    engine = ""
    try:
        with urllib.request.urlopen(
                urllib.request.Request("http://127.0.0.1:8000/v1/models"),
                timeout=4) as r:
            engine = json.load(r)["data"][0]["id"]
    except Exception:
        engine = "(engine not answering)"
    last = sh("journalctl", "--user", "-u", "spark-vllm-podman.service",
              "-n", "100", "--no-pager")
    interesting = [l for l in last.splitlines() if any(
        k in l for k in ("Resolved architecture", "Initializing a V1",
                         "GPU KV cache", "ERROR", "Exception"))]
    print(f"model.conf:   {conf.get('MODEL_ID', '?')} as "
          f"{conf.get('SERVED_MODEL_NAME', '?')}")
    print(f"service:      {unit} | watchdog: {wd}")
    if unit == "activating" or applying:
        print(f"switch:       IN PROGRESS -> {conf.get('MODEL_ID', '?')}")
    elif unit == "active":
        print(f"switch:       done - serving: {engine}")
    else:
        print(f"switch:       {unit} - engine: {engine}")
    print(f"live engine:  {engine}")
    ast = archive_call("--status")
    if ast.get("configured"):
        n = len(ast.get("archived") or [])
        mount = "mounted" if ast.get("mounted") else "NOT MOUNTED"
        free = human(ast.get("archive_free_bytes")) if ast.get("archive_free_bytes") else "?"
        print(f"archive:      {ast['archive_path']} ({mount}, {free} free, {n} archived)")
    else:
        print("archive:      not configured (--set-archive-path /archive)")
    if interesting:
        print("last engine:  " + interesting[-1].split("]: ")[-1][:110])


def main():
    if "--list" in sys.argv:
        run_list()
        return
    if "--status" in sys.argv:
        run_status()
        return
    if "--set-archive-path" in sys.argv:
        args = sys.argv[sys.argv.index("--set-archive-path") + 1:]
        if not args:
            print("usage: vllm-model-select --set-archive-path <dir>", file=sys.stderr)
            sys.exit(2)
        res = archive_call("--set-path", args[0], timeout=60)
        if res.get("ok"):
            print(f"archive path set: {res['path']} ({human(res.get('free_bytes'))} free)")
            sys.exit(0)
        print(f"error: {res.get('error')}", file=sys.stderr)
        sys.exit(2)
    for flag in ("--archive", "--unarchive"):
        if flag in sys.argv:
            args = sys.argv[sys.argv.index(flag) + 1:]
            if not args:
                print(f"usage: vllm-model-select {flag} <model-id>", file=sys.stderr)
                sys.exit(2)
            sys.exit(subprocess.run([sys.executable, ARCHIVER, flag, args[0]]).returncode)
    if "--selftest" in sys.argv:
        s = Selector()
        s.refresh()
        s.frame = urwid.Frame(urwid.AttrMap(s.body, "body"),
                              header=s.header(), footer=s.footer)
        s.loop = urwid.MainLoop(s.frame, s.palette)
        s.loop.start()
        s.loop.draw_screen()
        s.loop.stop()
        print("SELFTEST OK (rendered, exit clean)")
        return
    if "--delete" in sys.argv:
        print(delete_model(sys.argv[sys.argv.index("--delete") + 1]))
        sys.exit(0)
    if "--lock" in sys.argv:
        set_lock(sys.argv[sys.argv.index("--lock") + 1], True)
        print("locked")
        sys.exit(0)
    if "--unlock" in sys.argv:
        set_lock(sys.argv[sys.argv.index("--unlock") + 1], False)
        print("unlocked")
        sys.exit(0)
    Selector().run()


if __name__ == "__main__":
    main()
