# Model archive — moving checkpoints off the internal disk

The head has one 916 GB NVMe and four checkpoints fill ~507 GB of it
(157 GB DeepSeek ×2, 124 GB Qwen ×2). Every new model to experiment with has to
come from somewhere, so the cluster supports **archiving**: a checkpoint is
copied to an external disk, verified, and then removed from the internal cache.
The model's profile stays behind, so it remains listed, lockable and
re-selectable — it just cannot be selected until it is pulled back.

Two tools, one state:

- `vllm-model-archive.py` — the storage operations (CLI + `--json`).
- `vllm-model-select` — the TUI; `x` archive, `u` unarchive, `p` archive path.

## 1. The disk (one per node)

What we use: a 2 TB USB SSD per Spark (Fikwot FP80, USB 3.2 Gen2x2 enclosure,
rated ~2000 MB/s; the negotiated link depends on the port — check `lsusb -t`).
Capacity in practice: ~11–12 checkpoints of this class, or fewer if you keep
several DeepSeek-class models, since blobs hardlinked between repos are copied
as real bytes on the archive filesystem.

**ext4 is required.** The Hugging Face cache is built on symlinks
(`snapshots/<rev>/x -> ../../blobs/<sha>`), so exFAT/NTFS will not work. Keep the
cache itself on the internal NVMe — the archive disk only ever holds copies.

Prepare it on **each** node:

```bash
lsblk -o NAME,SIZE,MODEL,TRAN          # find the disk, e.g. sda (USB)
sudo wipefs -a /dev/sda                # only on a disk with nothing to keep
sudo mkfs.ext4 -m 0 -L archive /dev/sda   # -m 0: no 5% root reserve on a data disk
sudo mkdir -p /archive
sudo blkid -s UUID -o value /dev/sda   # note the UUID
```

Mount it via `/etc/fstab` so it survives reboots:

```
UUID=<uuid>  /archive  ext4  defaults,noatime,nofail,x-systemd.device-timeout=10  0  2
```

`nofail` is deliberate: an unplugged archive disk must not stop the head from
booting and serving. The tooling treats "not mounted" as a normal condition —
`--status` says so and `--archive` refuses instead of writing into an empty
mountpoint.

Then hand the mount to the service user (rootless podman + user units):

```bash
sudo chown kambiz:kambiz /archive
findmnt /archive && df -h /archive
```

## 2. Configure (per node)

```bash
vllm-model-archive --set-path /archive
vllm-model-archive --status
```

This writes `~/.config/spark-vllm/archive.conf`:

```
ARCHIVE_PATH=/archive
MIN_FREE_GB=10.0
```

`MIN_FREE_GB` is the safety margin kept free on the **internal** filesystem when
deciding whether an unarchive fits. The path must exist, be writable, and be on a
different filesystem than the cache — otherwise "archiving" would free nothing,
so it is refused.

The archive is **per node**: each Spark has its own disk and its own
`archive.conf`. The head tool never moves files on the worker. After
`deploy.sh --all` (which installs the helper on the worker too):

```bash
ssh gx10-02 vllm-model-archive --set-path /archive
ssh gx10-02 vllm-model-archive --list
```

## 3. Day to day

In the TUI (`vllm-model-select`):

| key | action |
|---|---|
| `x` | archive the highlighted model (needs a configured archive path) |
| `u` | unarchive the highlighted model (pre-checks free space) |
| `p` | set / change the archive path |

Rows show engine, state and size: cached models in normal colour, archived ones
**greyed out and marked `ARCHIVED`** with the archived size and `fits` /
`NO-SPACE`. A greyed row can be focused and unarchived, but not selected:
`Enter`, `s` and `A` refuse with "unarchive it first".

Archived rows stay visible because the profile is still valid: `flags/*.conf`,
`model.conf`, `locked.conf` and the lock list are untouched by archiving.

CLI, same operations:

```bash
vllm-model-select --archive     deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
vllm-model-select --unarchive   nvidia/Qwen3.8-Flash-Next-NVFP4
vllm-model-archive --list
vllm-model-archive --archive <id> --dry-run      # size/plan only
vllm-model-archive --unarchive <id> --dry-run    # fit check only
```

Long copies run in the background from the TUI (`~/.config/spark-vllm/archive.log`),
and the header shows the operation while it runs.

### Rules the tool enforces

- **Cannot archive the current boot model** (`model.conf`) — switch first.
- **Cannot archive a model the engine is serving right now** — stop or switch.
- **Unarchive is blocked when it would not fit**; the error lists the cached
  models to archive instead (largest wins first) and the exact shortfall.
- Locked models are not archived from the TUI (unlock with `l`).

## 4. What "archive" actually moves

For one model id, the payload is:

1. the repo directory `~/.cache/huggingface/hub/models--org--name/` (all
   snapshots, refs, per-repo blobs, symlinks preserved); and
2. every file outside that directory that its symlinks resolve to — Xet-backed
   weights live in `hub/blobs/<xx>/<sha>`, and those real bytes travel too, at
   their cache-relative paths.

Everything else is left alone: flags profiles, `model.conf`, `locked.conf`,
container images, the `spark-vllm-docker` fork, `~/miaai-qwen38`.

On disk:

```
$ARCHIVE_PATH/models/<safe-name>/
    manifest.json        model id, engine, host, sizes, timestamp, external list
    hub/                 cache-root-relative tree (restore = copy back into ~/.cache/huggingface/hub)
        models--org--name/...
        blobs/xx/<sha>   only the external blobs this model referenced
```

### Order of operations

1. Copy to the archive (rsync when available, internal copy otherwise).
2. **Verify** every entry: same type, same size, symlinks with the same target
   string.
3. Only then remove the internal repo dir, and remove external blobs **only** if
   no other cached model references them and their link count is 1.
4. Write `state: archived` in the manifest and report the freed bytes.

A failed or interrupted copy leaves the internal copy intact. A re-run resumes:
existing files of the same size are skipped.

Restoring is the mirror image, with one deliberate difference: no
`rsync --delete` is ever used against the cache root, because the blob store is
shared between models and a stray delete would take out another checkpoint.

## 5. Checking on it

```bash
vllm-model-archive --status     # path, mount state, free space, archived ids
vllm-model-archive --list       # per model: engine, state, cached/archive size, fits
./scripts/verify-deployment.sh  # includes the helper + archive status line
```

`vllm-model-select --status` also prints an `archive:` line, so the archive state
is part of the normal health view.

Both tools have synthetic tests that need no models and no archive disk (they
build throwaway cache trees under `$TMPDIR` and use `/dev/shm` as the "other
filesystem"):

```bash
python3 tests/test-archive.py
~/.local/share/vllm-tui/venv/bin/python tests/test-tui-archive.py   # needs urwid
```

## 6. Manual recovery

The archive layout is plain files — no tool needed to get data back:

```bash
# inspect
cat /archive/models/<safe-name>/manifest.json

# restore by hand (only the model's own paths; the tree is cache-root relative)
rsync -aH /archive/models/<safe-name>/hub/ ~/.cache/huggingface/hub/

# then re-run the TUI; the model shows as cached again
```

If a manifest is lost, the model can still be restored by hand from `hub/`; the
tool only uses the manifest to report state and sizes.
