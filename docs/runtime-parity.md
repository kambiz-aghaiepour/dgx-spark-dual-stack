# Runtime parity — repo vs live head

**Audited:** 2026-10-26 (initial audit) · **Repo changes:** 2026-10-27
**Reference head:** `gx10-01` — DGX OS, kernel 7.0.0-1019-nvidia, driver 580.178.04
**Repo HEAD at audit:** `f6eebd5` (2026-10-15)

This documents what in the repository is a faithful description of the deployed
cluster, what used to be local-only, and how to re-check the whole thing in one
command. The goal is **not** to change how the cluster runs — it is that a fresh
clone produces the same working layout, including the two pieces that used to
exist only on the head: the `urwid` venv wrapper and the `busy-loop-fix` mod.

Re-check at any time:

```bash
./scripts/verify-deployment.sh   # read-only parity check, exit 0 = in sync
vllm-model-select --status       # orchestration state + live engine probe
```

## What the checker covers

`scripts/verify-deployment.sh` compares the repo tree against the installed
copies (`cmp`, not hard-coded hashes), so it catches drift in either direction:

- the deployed scripts in `~/.local/bin` (runtime scripts, the selector TUI and
  `vllm-model-archive.py`) plus `~/spark-vllm-warmup.sh`
- every `config/flags/*` file in `~/.config/spark-vllm/flags/`
- the three `spark-vllm-*.service` units and `spark-vllm-podman.service.d/override.conf`
- the `vllm-model-select` launcher: must be the venv **wrapper**, not a symlink
  to the `.py`, with `urwid` importable in the venv it points at
- the launcher mods in `~/spark-vllm-docker/mods/` (`busy-loop-fix` byte-compared,
  `instanttensor-hybrid-draft-loader` presence-checked)
- the three units being `enabled`
- a live `vllm-model-select --status` run

## Audit method (2026-10-26)

The repo was cloned at `f6eebd5` and compared to the installed files on the head
with SHA256 and `cmp`, covering every script in `~/.local/bin`, the warmup
script, all flag files, and all systemd user units + `override.conf`.

Result: every file matched the repository except the `vllm-model-select`
launcher, which had no repository counterpart at all. The core scripts were last
changed in `5f77619` (2026-10-07); later commits were docs, flags and tooling, so
the repo was never stale about the code that actually runs.

Hashes of the runtime scripts at audit time (still current):

| File | SHA256 |
|---|---|
| `scripts/vllm-model-select.py` | `0bb7634a1fec770924727dceba0b928fda28b04ce2f8552f0ae2488a414e5128` |
| `scripts/vllm-model-set` | `36e19fa235e449c567a8957ce3e3d2fa506b0dd73d6a53abb0e1d5f8ba2fdfa2` |
| `scripts/spark-vllm-watchdog` | `6fca2e47a5fce0023c5493f8c575a50be9a3f4ba15cb9e32eb8649338b323e33` |
| `scripts/spark-vllm-podman-start-select` | `2c74eaf23ddc011b58b35629410ec5f078e3083c5b8d7f802469265a83373cc4` |
| `scripts/spark-vllm-podman-start` | `3952cd456d8ae5710050771e913991e8f84068dcbcc953d48024582517a5edbc` |
| `scripts/spark-vllm-podman-stop` | `26da487e7adeb62cc31189b91d813cecf4c48492449aa7fd55fcfe27c9753bff` |
| `scripts/spark-vllm-podman-health` | `1d343e2b561412f99da29197138348f763f5058fb53049a6c8b1b9fef15d55a3` |
| `scripts/spark-vllm-evict-cache.py` | `21169ac37d247aa523f7af78271cb824475e979334e8af82a65a2f7252d1b356` |
| `scripts/vllm-model-add.py` | `d2f8662d937cf8e140126aa99837460246416facc18f6cd68c0261eac40814b1` |
| `scripts/miaai_launch.sh` | `4cf2295d3385ec6caaf2c4940a2751d777e77f6cb50957561a24559d66965f62` |
| `scripts/spark-vllm-warmup.sh` | `29f35b59bae0af53d01230de69600a6f171333ef6acab4a2fb604ebd33820191` |
| `scripts/spark-stack-switch` | `c3296949a22e295b86bd4c68ae6423082dae99f104d4da40c59f08cf38cf02d9` |
| `scripts/spark-tensorfold-start` | `5e6b0a803b796d61b52e203ffa1b03a7b5f022fd65c0742e15f14081315fac36` |
| `scripts/spark-tensorfold-stop` | `40e63bfbdb4743fe2172e00af288b492996799f340b58811fc58a3cb26f38e77` |
| `scripts/spark-tensorfold-health` | `1c15af2c66eb4af0c28a08a0c03c686212a77fc74b72a47ab4279ff50ca5aa3b` |
| `scripts/spark-tensorfold-watchdog` | `c81b304c49047ea63ce19a199e0818cd1106be732299442f88f6a553a40d1edb` |
| `systemd/spark-tensorfold.service` | `6e2d079d98e1d29ec34a930233d7dad164f45930d1aff548bdbca46d66291a3f` |
| `systemd/spark-tensorfold-watchdog.service` | `d3fcc40c1c3d24e041eff4d0ce2b97b9e62846eee0cab43e33b9f2f3a75a7329` |

## Gaps and how the repo closes them

### 1. `vllm-model-select` launcher (was local-only)

Live head runs a shell wrapper around the identical Python file, because the TUI
imports `urwid` and system python does not have it:

```bash
#!/bin/bash
exec /home/kambiz/.local/share/vllm-tui/venv/bin/python /home/kambiz/.local/bin/vllm-model-select.py "$@"
```

`deploy.sh` used to create `ln -sf vllm-model-select.py vllm-model-select`, which
resolves the `#!/usr/bin/env python3` shebang to system python — so a fresh deploy
broke the command, and re-running the old installer on the head also broke it by
replacing the wrapper.

Now:

- `scripts/vllm-model-select` — the wrapper (uses `$HOME`, checks for the venv
  interpreter and the selector, clear error if either is missing).
- `scripts/setup-tui-venv.sh` — creates `~/.local/share/vllm-tui/venv` with
  `urwid==4.2.5`; falls back to `uv` when `python3 -m venv`/`pip` is unavailable.
- `deploy.sh` calls the bootstrap, then installs the wrapper — and removes any
  legacy symlink first, because `install` would otherwise follow it and overwrite
  `vllm-model-select.py` itself.

### 2. `busy-loop-fix` mod (was an untracked directory in the docker fork)

`spark-vllm-podman-start` applies both `mods/instanttensor-hybrid-draft-loader`
(tracked in the `spark-vllm-docker` fork) and `mods/busy-loop-fix` (owned by this
repo at `scripts/mods/busy-loop-fix`, previously never copied to where the
launcher looks for it). A fresh deploy would fail at launch on the missing mod,
and `git clean -fd` in the fork silently deleted it from a working head.

`deploy.sh` now installs both: it hard-fails with a clear message if the fork or
`instanttensor-hybrid-draft-loader` is missing, then `cp -a`'s `busy-loop-fix`
into `$SPARK_VLLM_DOCKER/mods/` (default `~/spark-vllm-docker`). `--skip-mods`
keeps the old scripts-only behaviour for a host that has no docker fork yet.

### 3. `deploy.sh` overwrote the live `model.conf`

The installer unconditionally replaced `~/.config/spark-vllm/model.conf` with
`config/model.conf.template`, which would silently reset the boot default on an
existing head.

Now the template is only written when no `model.conf` exists; otherwise it is
kept and the run says so. `--force-config` restores the old replace behaviour
when a reset is actually wanted.

### 4. `switch.sh` was referenced by the docs but not installed

`docs/setup-fresh-install.md` used `./scripts/switch.sh` (works from a checkout),
but the live head had no `~/.local/bin/switch.sh`. Chosen fix: install it.
`deploy.sh` copies it with the other scripts; it is a thin wrapper over
`vllm-model-select --status/--list` and `vllm-model-set`, so it adds no runtime
dependency and does not affect the running service. `verify-deployment.sh`
reports it as `WARN` (not a failure) when absent, so a head that was not
re-deployed still reports parity.

### 5. Runtime state vs repository artifacts

These are **not** repo artifacts and must not be copied between hosts:

| Path | What it is |
|---|---|
| `~/.config/spark-vllm/model.conf` | current selection; written by `vllm-model-set` |
| `~/.config/spark-vllm/model.conf.bak` | local backup taken by `vllm-model-set` |
| `~/.config/spark-vllm/locked.conf` | local lock list (model registry) |
| `~/.config/spark-vllm/flags/` | **tracked** configuration (per-model flags/env) |
| `~/spark-vllm-docker` | separate fork of `eugr/spark-vllm-docker`; supplies `launch-cluster.sh` + the tracked `instanttensor-hybrid-draft-loader` mod |
| `~/miaai-qwen38` | produced by `podman-adapt-runbook.sh`; not part of this repo |
| `~/.local/share/vllm-tui/venv` | selector tooling venv; created by `setup-tui-venv.sh` |

The worker (`gx10-02`) needs no orchestration scripts: the head drives the
launch, and the worker only needs the container runtime, the model files and the
runtime helpers copied by `deploy.sh --all`.

## Archive tool (added 2026-10-27)

`vllm-model-archive.py` plus the `x` / `u` / `p` keys in the selector let a
checkpoint move to an external ext4 disk and back, so several models can be kept
without several hundred GB of internal disk. The operation spans **both nodes**
(the pair is tensor-parallel, so a model must be cached on each), and every size
is reported per node. Procedure and layout: [docs/archive.md](archive.md).

Parity notes:

- `deploy.sh` installs the helper as an executable (the head runs it over ssh);
  `--all` copies it to the worker, which has its own disk and its own
  `ARCHIVE_PATH`.
- `verify-deployment.sh` prints the archive state per node, so a worker without
  the helper or without a configured/mounted path shows up as such.
- The helper is reported as `WARN`, not `FAIL`, when missing: a head that has not
  been re-deployed keeps working, with archive/unarchive disabled.
- Archiving never touches the parity-checked configuration: flags profiles,
  `model.conf`, `locked.conf` and the units stay byte-identical; only the HF cache
  tree moves. `verify-deployment.sh` therefore still passes on a head with
  archived models.

## Reproducing from scratch

```bash
git clone https://github.com/kambiz-aghaiepour/spark-vllm-docker.git ~/spark-vllm-docker
git clone https://github.com/kambiz-aghaiepour/dgx-spark-dual-stack.git
cd dgx-spark-dual-stack
./scripts/deploy.sh              # scripts + venv wrapper + flags + units + mods
./scripts/verify-deployment.sh   # must exit 0
vllm-model-select --status
```

On an **existing** head, the live runtime already matched the core scripts, so
re-running `deploy.sh` is not required for the cluster to serve. It is required
only to pick up the new wrapper/`switch.sh`/mods placement — and it is now
non-destructive to `model.conf`.

Back up first if you want belt-and-braces:

```bash
cp -a ~/.config/spark-vllm/model.conf ~/.config/spark-vllm/model.conf.pre-parity
cp -a ~/.config/spark-vllm/locked.conf ~/.config/spark-vllm/locked.conf.pre-parity
cp -a ~/.local/bin/vllm-model-select ~/.local/bin/vllm-model-select.pre-parity
```

## Acceptance criteria

- `verify-deployment.sh` exits 0 on the head.
- `vllm-model-select --status` prints `model.conf`, `service`, `watchdog` and
  `live engine` without a traceback.
- `systemctl --user is-enabled spark-vllm-podman spark-vllm-health spark-vllm-warmup
  spark-tensorfold spark-tensorfold-watchdog` reports all five as `enabled`.
- `spark-stack-switch --status` names the stack in `~/.config/spark-vllm/stack.conf` and
  says which unit and API belong to each stack; the stack that is not switched in reports
  `inactive`.
- `~/spark-vllm-docker/mods/busy-loop-fix` and `.../instanttensor-hybrid-draft-loader`
  both exist.
- A fresh head can run `deploy.sh` and start the service with no manual copying.
