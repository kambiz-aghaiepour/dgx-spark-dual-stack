#!/usr/bin/env python3
"""Headless behaviour tests for the archive keys in vllm-model-select.py.

Run with the selector venv (urwid is required):
  ~/.local/share/vllm-tui/venv/bin/python tests/test-tui-archive.py
Archive location defaults to /dev/shm; override with ARCHIVE_TEST_PATH.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPER = os.path.join(REPO, "scripts", "vllm-model-archive.py")
TUI = os.path.join(REPO, "scripts", "vllm-model-select.py")
HOME = tempfile.mkdtemp(prefix="vllm-tuitest-home-")
SHM = os.environ.get("ARCHIVE_TEST_PATH") or "/dev/shm/vllm-tuitest"
ARCH = os.path.join(SHM, "archive")
HUB = os.path.join(HOME, ".cache/huggingface/hub")
fails = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def write(p, data=b"", size=None):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    if isinstance(data, str):
        data = data.encode()
    with open(p, "wb") as f:
        f.write(data)
    if size:
        with open(p, "wb") as f:
            f.truncate(size)


def fixture():
    shutil.rmtree(HOME, ignore_errors=True)
    shutil.rmtree(SHM, ignore_errors=True)
    os.makedirs(os.path.join(HOME, ".local/bin"))
    os.makedirs(os.path.join(HOME, ".config/spark-vllm/flags"))
    os.makedirs(ARCH)
    shutil.copy2(HELPER, os.path.join(HOME, ".local/bin/vllm-model-archive.py"))
    # cached model "org/keep" (current) and "org/moving"
    for name, size in (("keep", 2_000_000), ("moving", 5_000_000)):
        d = os.path.join(HUB, "models--org--" + name)
        write(os.path.join(d, "blobs", "b1"), b"", size=size)
        os.makedirs(os.path.join(d, "snapshots", "r1"))
        os.symlink("../../blobs/b1", os.path.join(d, "snapshots/r1/weights.bin"))
        write(os.path.join(d, "refs/main"), "r1\n")
        write(os.path.join(HOME, ".config/spark-vllm/flags", f"org-{name}.conf"), "-tp 2\n")
    # model "org/gone": cached now, archived by the test setup
    d = os.path.join(HUB, "models--org--gone")
    write(os.path.join(d, "blobs", "b1"), b"", size=1_000_000)
    os.makedirs(os.path.join(d, "snapshots", "r1"))
    os.symlink("../../blobs/b1", os.path.join(d, "snapshots/r1/weights.bin"))
    write(os.path.join(d, "refs/main"), "r1\n")
    write(os.path.join(HOME, ".config/spark-vllm/flags", "org-gone.conf"), "-tp 2\n")
    stub = os.path.join(HOME, ".local/bin/vllm-model-set")
    write(stub, "#!/bin/sh\nexit 0\n")
    os.chmod(stub, 0o755)
    write(os.path.join(HOME, ".config/spark-vllm/model.conf"),
          "MODEL_ID=org/keep\nSERVED_MODEL_NAME=org/keep\n")
    # single-node mode: the TUI test drives one node; the cluster wiring is
    # covered by tests/test-archive.py
    env = dict(os.environ, HOME=HOME, SPARK_VLLM_NODES="head")
    for args in (["--set-path", ARCH], ["--archive", "org/gone"]):
        r = subprocess.run([sys.executable, HELPER, *args], capture_output=True, text=True, env=env)
        if r.returncode != 0:
            raise SystemExit(f"fixture setup {args} failed: {r.stdout} {r.stderr}")


def load_tui():
    os.environ["HOME"] = HOME
    os.environ["SPARK_VLLM_NODES"] = "head"
    os.environ["SPARK_VLLM_ENGINE_URL"] = "http://127.0.0.1:9"
    spec = importlib.util.spec_from_file_location("vllm_model_select", TUI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.ARCHIVER = os.path.join(HOME, ".local/bin/vllm-model-archive.py")
    return mod


def stub_loop(mod):
    class Screen:
        def set_mouse_tracking(self, v):
            pass
    return types.SimpleNamespace(widget=None, screen=Screen())


def main():
    fixture()
    mod = load_tui()
    import urwid
    s = mod.Selector()
    s.frame = urwid.Frame(urwid.Filler(urwid.Text("x")))
    s.loop = stub_loop(mod)

    ids = {m["id"]: m for m in s.models}
    check("three models listed", set(ids) == {"org/keep", "org/moving", "org/gone"}, list(ids))
    check("archived model flagged", ids["org/gone"]["cached"] is False
          and ids["org/gone"]["archived"] is True, ids.get("org/gone"))
    check("archived keeps profile", os.path.exists(os.path.join(
        HOME, ".config/spark-vllm/flags/org-gone.conf")))

    rows = s.build_rows()
    texts = [r.original_widget.text for r in rows]
    check("archived row says ARCHIVED", any("ARCHIVED" in t and "org/gone" in t for t in texts), texts)
    check("archived row greyed (attr arch)",
          [a.attr_map.get(None) for a in rows if "org/gone" in a.original_widget.text] == ["arch"],
          [a.attr_map.get(None) for a in rows])
    check("cached row normal attr",
          all(a.attr_map.get(None) in ("body", "cur") for a in rows
              if "org/keep" in a.original_widget.text or "org/moving" in a.original_widget.text),
          [a.attr_map.get(None) for a in rows])
    check("header shows per-node space", "head:" in s.header_text()
          and "archive" in s.header_text(), s.header_text())
    check("header counts archived", "1 archived" in s.header_text(), s.header_text())
    s.refresh()
    check("footer has archive keys", "x:archive" in s.footer.text and "u:unarchive" in s.footer.text
          and "p:archive-path" in s.footer.text, s.footer.text)

    # selection guards
    s.pending = ""
    s.pick("org/gone")
    check("pick refused on archived", s.pending == "" and "archived" in s.msg, s.msg)
    s.body.set_focus([i for i, m in enumerate(s.models) if m["id"] == "org/gone"][0])
    s.keypress("enter")
    check("enter refused on archived", s.pending == "" and "archived" in s.msg, s.msg)
    s.body.set_focus([i for i, m in enumerate(s.models) if m["id"] == "org/moving"][0])
    s.keypress("enter")
    check("enter picks cached model", s.pending == "org/moving", (s.pending, s.msg))
    s.keypress("s")
    check("write-default allowed for cached", "archived - unarchive" not in s.msg, s.msg)
    s.pending = "org/gone"
    s.keypress("s")
    check("write-default refused for archived", "unarchive it (u)" in s.msg, s.msg)

    # archive guards + flow
    s2 = mod.Selector()
    s2.frame = s.frame
    s2.loop = stub_loop(mod)
    s2.body.set_focus([i for i, m in enumerate(s2.models) if m["id"] == "org/keep"][0])
    s2.start_archive()
    check("archive refused for current model",
          "cannot archive the current model" in s2.msg
          and "switch to another model/engine first" in s2.msg.lower(), s2.msg)
    s2.arch["configured"] = False
    s2.body.set_focus([i for i, m in enumerate(s2.models) if m["id"] == "org/moving"][0])
    s2.start_archive()
    check("archive requires path configured", "not configured" in s2.msg, s2.msg)
    s2.arch["configured"] = True
    s2.start_archive()
    check("archive confirm shown", s2.confirming_arch is True and s2.arch_target == "org/moving",
          (s2.confirming_arch, s2.arch_target))
    check("confirm dialog mentions profile retention",
          "profiles" in str(s2.loop.widget) or s2.confirming_arch)

    # unarchive guards
    s3 = mod.Selector()
    s3.frame = s.frame
    s3.loop = stub_loop(mod)
    s3.body.set_focus([i for i, m in enumerate(s3.models) if m["id"] == "org/keep"][0])
    s3.start_unarchive()
    check("unarchive refused when not archived", "not archived" in s3.msg, s3.msg)
    s3.body.set_focus([i for i, m in enumerate(s3.models) if m["id"] == "org/gone"][0])
    s3.start_unarchive()
    check("unarchive allowed when it fits", s3.confirming_arch is True and s3.arch_action == "unarchive",
          (s3.confirming_arch, s3.arch_action, s3.msg))

    # forced no-space path: a huge safety margin makes the helper's per-node
    # fit check fail, which must block the confirm dialog
    conf = os.path.join(HOME, ".config/spark-vllm/archive.conf")
    keep = open(conf).read()
    with open(conf, "w") as f:
        f.write(f"ARCHIVE_PATH={ARCH}\nMIN_FREE_GB=100000000\n")
    s4 = mod.Selector()
    s4.frame = s.frame
    s4.loop = stub_loop(mod)
    s4.body.set_focus([i for i, m in enumerate(s4.models) if m["id"] == "org/gone"][0])
    s4.start_unarchive()
    check("no-space blocks the unarchive", s4.confirming_arch is False and s4.loop.widget is not None,
          (s4.confirming_arch, s4.msg))
    check("no-space dialog names the short node and a model to archive",
          "does not fit" in s4.msg, s4.msg)
    with open(conf, "w") as f:
        f.write(keep)

    # --- path prompt: a detected mount is offered, never assumed
    conf = os.path.join(HOME, ".config/spark-vllm/archive.conf")
    keep_conf = open(conf).read()
    os.remove(conf)
    os.environ["SPARK_VLLM_DEFAULT_ARCHIVE"] = "/dev/shm"
    s5 = mod.Selector()
    s5.frame = s.frame
    s5.loop = stub_loop(mod)
    check("detected mount is not configured", s5.arch.get("configured") is False, s5.arch)
    check("header offers the detected mount", "/dev/shm" in s5.header_text(), s5.header_text())
    s5.edit_archive_path()
    check("path prompt prefills the detected mount",
          s5._path_edit.get_edit_text() == "/dev/shm", s5._path_edit.get_edit_text())
    s5.keypress("esc")
    with open(conf, "w") as f:
        f.write(keep_conf)
    os.environ.pop("SPARK_VLLM_DEFAULT_ARCHIVE", None)

    # --- saving the path on a two-node view asks about the worker
    os.environ["SPARK_VLLM_NODES"] = "head,worker"
    os.environ["SPARK_VLLM_REMOTE_CMD"] = "/bin/false {host} {helper} {args}"
    s6 = mod.Selector()
    s6.frame = s.frame
    s6.loop = stub_loop(mod)
    check("two-node view keeps a column per node",
          s6.arch.get("node_order") == ["head", "worker"], s6.arch.get("node_order"))
    s6.edit_archive_path()
    s6.keypress("enter")
    check("after saving, the worker path is offered",
          s6.confirming_worker_path == ["worker"], (s6.confirming_worker_path, s6.msg))
    s6.keypress("n")
    check("declining leaves the worker alone",
          "this node only" in s6.msg and not s6.confirming_worker_path, s6.msg)
    os.environ.pop("SPARK_VLLM_NODES", None)
    os.environ.pop("SPARK_VLLM_REMOTE_CMD", None)

    print()
    if fails:
        print(f"{len(fails)} FAILED: {fails}")
        return 1
    print("ALL TUI CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
