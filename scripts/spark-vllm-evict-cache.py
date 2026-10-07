#!/usr/bin/env python3
"""spark-vllm-evict-cache - release a checkpoint's clean pages (GB10 unified
memory). POSIX_FADV_DONTNEED on every regular file under a HF hub model dir.
Usage: spark-vllm-evict-cache.py <hub-model-dir>
"""
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/.cache/huggingface/hub")
n = 0
total = 0
for dirpath, _dirs, files in os.walk(root):
    for f in files:
        p = os.path.join(dirpath, f)
        try:
            fd = os.open(p, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
            n += 1
        except OSError:
            pass
print(f"evicted {n} files under {root}")
