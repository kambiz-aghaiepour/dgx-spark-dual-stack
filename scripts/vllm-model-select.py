#!/usr/bin/env python3
"""vllm-model-select - Phase 2 TUI (urwid) for the DGX cluster.

Lists models cached on the head node, pick the boot default and a
served-model-name alias, write model.conf, or apply via vllm-model-set.

Non-interactive: vllm-model-select --list
"""
import glob
import os
import subprocess
import sys

import urwid

CONF = os.path.expanduser("~/.config/spark-vllm/model.conf")
HUB = os.path.expanduser("~/.cache/huggingface/hub")
WORKER = "gx10-02"
PATH_SETTER = os.path.expanduser("~/.local/bin/vllm-model-set")
LOCKS = os.path.expanduser("~/.config/spark-vllm/locked.conf")
FLAGS_DIR = os.path.expanduser("~/.config/spark-vllm/flags")


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
    conf = read_conf()
    cur = conf.get("MODEL_ID", "")
    print(f"{'MODEL':62} {'size(worker)':>13}  status")
    for m in scan_models():
        mark = "*" if m["id"] == cur else " "
        print(f"{mark} {m['id']:<61} {m['worker'] or 'n/a':>13}  "
              f"{'current' if m['id'] == cur else ''}")
    print(f"\ncurrent: {cur} -> {conf.get('SERVED_MODEL_NAME', '')}")


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
    ]

    def __init__(self):
        self.conf = read_conf()
        self.models = scan_models()
        self.pending = self.conf.get("MODEL_ID", "")
        self.alias = self.conf.get("SERVED_MODEL_NAME", "")
        self.msg = ""
        self.editing = False
        self.picked = ""
        self.capture = True
        self.confirming = False
        self.confirming_del = False
        self.del_target = ""
        self.exec_apply = None
        self.body = urwid.ListBox(urwid.SimpleListWalker([]))
        self.footer = urwid.Text("", wrap="clip")

    def pick(self, mid):
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
            rows.append(urwid.AttrMap(
                PickRow(f"{curm}{mark} {m['id']:<52} {m['worker'] or 'n/a':>10}"
                        f"{'  CURR' if m['id'] == cur else ''}{locked}", self, m["id"]),
                "cur" if m["id"] == self.pending else "body", "sel"))
        return rows

    def refresh(self):
        self.body.body[:] = self.build_rows()
        self.footer.set_text(
            f"{self.msg}  ||  alias: {self.alias}  click/Enter:pick  "
            f"a:alias(ctrl-u=clear)  m:mouse  d:delete(confirm)  "
            f"l:lock/unlock  s:write-default  A:apply(~5-10 min)  "
            f"r:refresh  q:quit")

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
        base = (f" vllm-model-select | current: {self.conf.get('MODEL_ID', '')} "
                f"as {self.conf.get('SERVED_MODEL_NAME', '')} | "
                f"{len(self.models)} cached")
        st = self.switch_status()
        return base + (f"\n {st}" if st else "")

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

    def keypress(self, key):
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
            idx = self.body.focus_position
            if self.models:
                self.pending = self.models[idx]["id"]
                self.msg = f"pending: {self.pending}"
                self.refresh()
            return True
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
        self.header()
        loop.set_alarm_in(10, self._tick)

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
    if interesting:
        print("last engine:  " + interesting[-1].split("]: ")[-1][:110])


def main():
    if "--list" in sys.argv:
        run_list()
        return
    if "--status" in sys.argv:
        run_status()
        return
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
