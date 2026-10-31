# Model archive — moving checkpoints off the internal disks

A checkpoint only serves if it is in **both** nodes' caches (the pair runs
tensor-parallel), so archive and unarchive are **cluster operations**: every node
moves its own copy to its own archive disk, and every size is reported per node.
One node done and the other not would leave a model that cannot start while
still occupying space — the tool never does that.

Two tools, one state:

- `vllm-model-archive.py` — storage operations (CLI + `--json`), orchestrating
  the worker over ssh from the head.
- `vllm-model-select` — the TUI; `x` archive, `u` unarchive, `p` archive path.

## The default follows the engine, not vLLM

`stack.conf` is the single selection:

```
STACK=vllm|tensorfold
TF_PROFILE=mia-dsv41
VLLM_MODEL=<hf model id>
```

`VLLM_MODEL` is the default only while `STACK=vllm`. Under TensorFold it is a parked
choice, so its cache can be archived; it must be unarchived before switching back to vLLM.
`spark-stack-switch` checks both nodes and refuses to stop TensorFold when the target vLLM
checkpoint is not resident. A default is always on internal NVMe; the archive disk only
holds restorable copies and never bootstraps an engine.

## 1. The disks (one per node)

What we use: a 2 TB USB SSD per Spark (Fikwot FP80, USB 3.2 Gen2x2 enclosure,
rated ~2000 MB/s; the negotiated link depends on the port — check `lsusb -t`).
Capacity in practice: ~11–12 checkpoints of this class, fewer if you keep several
DeepSeek-class models.

**ext4 is required.** The Hugging Face cache is built on symlinks
(`snapshots/<rev>/x -> ../../blobs/<sha>`), so exFAT/NTFS will not work. Keep the
cache itself on the internal NVMe — the archive disk only ever holds copies.

Prepare it on **each** node (the disks are per node; they are not shared):

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

`nofail` is deliberate: an unplugged archive disk must not stop a node from
booting and serving. The tooling treats "not mounted" as a normal condition —
`--status` says so and `--archive` refuses instead of writing into an empty
mountpoint.

Then hand the mount to the service user (rootless podman + user units):

```bash
sudo chown kambiz:kambiz /archive
findmnt /archive && df -h /archive
```

## 2. Configure (each node separately)

Configuration stays a per-node decision: **a mounted disk is not a configured
node.** The tool checks `archive.conf`, not the mount, so a fresh `/archive`
that nobody enabled still refuses to archive — that is the prerequisite.

In the TUI, `p` does the whole job:

1. The prompt is prefilled with this node's path, or — if nothing is configured
   and `/archive` is detected as a real mount here (a separate filesystem from
   the cache, writable) — with `/archive`, titled
   `"/archive is mounted here - enter to enable"`.
2. On save it writes this node's `archive.conf`.
3. On a multi-node view it then asks
   `"Also set the archive path on worker?"`, showing the path. `y` runs
   `--set-path … --node worker` over ssh; `n` (the default) leaves the worker
   untouched and says so.

Detection only ever prefills the prompt: the header shows
`/archive detected - p to enable` until you actually press `p`, and
`--status` prints `/archive is mounted there; enable it with p`.

From the shell — same operations, still one node unless you ask for more:

```bash
vllm-model-archive --set-path /archive                 # this node
vllm-model-archive --set-path /archive --node worker   # the worker, over ssh
vllm-model-archive --set-path /archive --node all      # both, reporting each
vllm-model-archive --status                            # a line per node
```

On the worker the helper has to exist first — `./scripts/deploy.sh --all` copies
`vllm-model-archive.py` to `~/.local/bin` there and makes it executable (the head
runs it over ssh). `--set-path --node worker` fails loudly, naming the node, if
the directory is missing, not writable, or on the same filesystem as that
node's cache.

`deploy.sh` also writes `~/.config/spark-vllm/node.conf` (`NODE=head` here,
`NODE=worker` on the worker) so a helper started on the worker knows it is the
worker; without that it calls itself head and "fails to reach" itself.

Each node writes its own `~/.config/spark-vllm/archive.conf`:

```
ARCHIVE_PATH=/archive
MIN_FREE_GB=10.0
```

`MIN_FREE_GB` is the safety margin kept free on that node's **internal**
filesystem when deciding whether an unarchive fits. The path must exist, be
writable, and be on a different filesystem than the cache — otherwise
"archiving" would free nothing, so it is refused.

Archive paths may differ per node; nothing assumes they match.

## 3. Day to day

In the TUI (`vllm-model-select`) the list has **one size column per node**:

```
MODEL                        ENGINE STATE        HEAD    WORKER  status
deepseek-ai/DeepSeek-V4-...  b12x   cached     167.8 GB 167.8 GB
drowzeys/keys-Qwen3.8-...    miaai  cached     132.7 GB 124.3 GB  current live
```

| key | action |
|---|---|
| `x` | archive on every node (needs a configured path on each) |
| `u` | unarchive on every node (checks each node's free space) |
| `p` | set / change this node's archive path |

The confirm dialogs show the per-node numbers before anything is copied: bytes to
move, estimated bytes freed, and the archive free space on each node — and for an
unarchive, what each node needs and what it would have left.

Rows are `cached` (present on every node), `archived` (present on none), or
`PARTIAL` (the cluster is half-archived — e.g. after a manual intervention); an
archived row is **greyed out and unselectable** until unarchived, but stays
listed because its profile is still valid.

CLI, same operations:

```bash
vllm-model-archive --archive   <id>            # all nodes
vllm-model-archive --archive   <id> --dry-run  # per-node plan, nothing moved
vllm-model-archive --unarchive <id>            # all nodes, per-node fit check
vllm-model-archive --node head  --archive <id> # one node only (debugging)
vllm-model-select --list
```

Long copies run in the background from the TUI
(`~/.config/spark-vllm/archive.log`); the header shows the operation, the footer
the per-node result.

### Rules the tool enforces

- **The active model cannot be archived** — neither the boot default in
  `model.conf` nor whatever the engine is serving: "cannot archive the current
  model … switch to another model/engine first". The check runs on both nodes and
  before any copy.
- **Nothing is deleted until every node's copy verifies.** A failure during the
  copy phase rolls the staged archive directories back, so you never get a
  half-archived model.
- **Unarchive is refused if any node cannot fit it**, naming the node(s), the
  shortfall on each, and which models could be archived to make room (per node).
- Locked models are not archived from the TUI (unlock with `l`).

## 4. What "archive" actually moves (per node, and it differs)

For one model id, the payload on a node is:

1. the repo directory `~/.cache/huggingface/hub/models--org--name/` (snapshots,
   refs, per-repo blobs, symlinks preserved); and
2. every file outside it that its symlinks resolve to — on the head the weights
   are Xet-backed and live in `hub/blobs/<xx>/<sha>`, shared between models.

The two nodes are **not** identical, which is exactly why sizes are per node:

- **head**: copied 489 GB of blobs shared across four models. Archiving one Qwen
  twin frees ~75 GB, not the 133 GB it "occupies", because 57 GB is still
  referenced by the other twin. The tool estimates the real win by excluding
  blobs another cached model points at.
- **worker**: each model is fully materialized inside its own repo dir
  (124/124/157/157 GB, no sharing), so archiving it frees the whole copy.

Everything else is untouched: flags profiles, `model.conf`, `locked.conf`,
container images, the `spark-vllm-docker` fork, `~/.cache/vllm` (compile/autotune
caches, 131 MB) and `~/miaai-qwen38` (head-only patched modules).

On each node's archive disk:

```
$ARCHIVE_PATH/models/<safe-name>/
    manifest.json        model id, engine, node, sizes, state, external list
    hub/                 cache-root-relative tree (restore copies it back)
        models--org--name/...
        blobs/xx/<sha>   only the external blobs this model referenced
```

### Order of operations (cluster)

1. **Plan** on every node: refuse an active model, missing/unmounted path, too
   little archive space. Reports per-node bytes, estimated freed bytes, free
   space.
2. **Copy + verify** on every node (`state: staged` in each manifest). Nothing is
   deleted in this phase; a failure here rolls back the nodes already staged.
3. **Commit** on every node: remove the internal repo dir, remove external blobs
   only if no other cached model references them and `nlink == 1`, then mark
   `state: archived` and record that node's freed bytes.
4. Report: bytes copied and bytes freed, per node and totalled.

A crash between phases leaves `staged` copies plus intact sources; re-running the
same command finishes the job (a staged manifest whose source is already gone is
finalized instead of failing). Restores are additive and resumable: existing
files of the same size are skipped, and `rsync --delete` is never used against
the cache root because the blob store is shared.

## 5. Checking on it

```bash
vllm-model-archive --status     # per node: path, mount, archive free, cache free
vllm-model-archive --list       # per model: state + a size column per node
./scripts/verify-deployment.sh  # helper + archive status
vllm-model-select --status      # archive[head]: …, archive[worker]: …
```

Both tools have synthetic tests that need no models, no disks and no ssh. The
cluster test stands a second HOMEs in for the worker and wires it in through
`SPARK_VLLM_REMOTE_CMD`, so the orchestrator still crosses a real subprocess
boundary:

```bash
python3 tests/test-archive.py
~/.local/share/vllm-tui/venv/bin/python tests/test-tui-archive.py   # needs urwid
```

### Knobs (environment)

| variable | meaning |
|---|---|
| `WORKER` | ssh alias of the second node (default `gx10-02`) |
| `SPARK_VLLM_NODES` | node names in order (default `head,worker`) |
| `SPARK_VLLM_NODE` | name of *this* machine — set by the remote runner, not by hand |
| `SPARK_VLLM_REMOTE_CMD` | transport template (`ssh … {host} {helper} {args}`) |
| `SPARK_VLLM_ARCHIVE_HELPER` | helper path used for the remote call |
| `SPARK_VLLM_ENGINE_URL` | engine probe used to detect the served model |

## 6. Manual recovery

The archive layout is plain files — no tool needed to get data back:

```bash
# inspect (same layout on both nodes)
cat /archive/models/<safe-name>/manifest.json

# restore by hand on that node (cache-root relative)
rsync -aH /archive/models/<safe-name>/hub/ ~/.cache/huggingface/hub/

# on the head only, for the record: repeat on the worker if you restore there
```

If a manifest is lost, the model can still be restored by hand from `hub/`; the
tool only uses the manifest to report state and sizes.
