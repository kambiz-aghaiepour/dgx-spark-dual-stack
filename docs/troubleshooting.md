# Troubleshooting — observations from the field

## 1. FlashInfer autotune stall (b12x engine)

Symptom (journal):

```
INFO [shm_broadcast.py:813] No available shared memory broadcast block found in 60 sec
```

For **10+ min with no other progress** (no `blob data` / `Warmed` /
`Ready` lines), the worker is hung in FlashInfer autotune:

```
INFO [kernel_warmup.py:350] Running FlashInfer autotune with ... tokens
```

This is **flaky — not config-dependent**: it hit plain *and* tuned b12x
profiles on this kit (2 stalls in one day). The *same warning* is benign when
progress continues (worker compiling/autotuning is "time-consuming work" the
message describes) — check for nearby progress markers first (5 min with ≥1
marker = fine).

Mitigations:
- DeepSeek's b12x path uses B12X attention — it **skips this autotune
  entirely** (why it never stalls; it's the production default).
- flash-next now runs on the native (MiaAI) engine, whose warmup has never
  stalled here.
- If you must run flash-next on the b12x engine: one clean retry
  (`systemctl --user stop; podman rm -a -f; start`).

## 2. conmon teardown noise

```
conmon ... <nwarn>: Failed to write to remote console socket
```

Benign: emitted during container kill/rm (the console socket's far end
closed). Appears next to `Failed with result 'signal'`. Not a failure.

## 3. Page-cache OOM race (GB10 unified memory)

Loading a ~124 GB checkpoint with page-cache still holding it can push the
engine into startup OOM. `vllm-model-set --apply` evicts the checkpoint from
cache on both nodes before start (`spark-vllm-evict-cache.py`). If you see a
startup OOM, run the evict + retry:

```bash
python3 ~/.local/bin/spark-vllm-evict-cache.py ~/.cache/huggingface/hub/models--<id>--*
```

## 4. Reasoning-model probe nuance

flash-next is a reasoning model: with `max_tokens:1` the completion can return
`content: null` (all budget consumed in the reasoning phase) — **HTTP 200 +
valid JSON**, so watchdog/health pass. Only the *bar* semantics change for
manual tests (use max_tokens ≥ 32).

## 5. Slow/failed stop (deactivating for minutes)

After a stalled engine, `systemctl --user stop` can sit in `deactivating`
a long time (engine wrapper doesn't exit promptly). Force:

```bash
systemctl --user kill spark-vllm-podman.service
podman kill -a; podman rm -a -f
ssh gx10-02 "podman kill -a; podman rm -a -f"
```

## 6. Worker has no internet

- Image distribution: `podman pull` on head, then
  `podman save <image> | ssh gx10-02 podman load` (the runbook's flow does this).
- Weights: rsync from head `~/.cache/huggingface/`.

## 7. `docker: permission denied ... docker.sock`

Normal. This setup uses **rootless podman**; Docker CLI may exist but its
daemon socket is root-owned. Always: `export XDG_RUNTIME_DIR=/run/user/$(id -u)`
before `podman`/`systemctl --user` in scripts.
