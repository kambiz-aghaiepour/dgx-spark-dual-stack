#!/usr/bin/env python3
"""Sustained output-tok/s ceiling test for the DGX vLLM cluster.

Keeps N concurrent requests in flight for DUR seconds; each request uses a
decode-dominated prompt (~1.2K tokens) and max_tokens=1024, greedy.
Aggregate output tokens/s is computed from the API's own usage fields.
"""
import json, sys, threading, time, urllib.request

URL = "http://gx10-01:8000/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"
PROMPT = "Write the numbers from 1 to 1500, one per line. Start with 1 and go in order without stopping."

def worker(stop, results):
    while not stop.is_set():
        p = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
             "max_tokens": 1024, "temperature": 0.0,
             "chat_template_kwargs": {"thinking": False}}
        t0 = time.time()
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(
                URL, data=json.dumps(p).encode(),
                headers={"Content-Type": "application/json"}), timeout=900))
            t1 = time.time()
            u = d["usage"]
            results.append({"t0": t0, "dt": t1 - t0, "pt": u["prompt_tokens"],
                            "ct": u["completion_tokens"]})
        except Exception as e:
            results.append({"t0": t0, "dt": time.time() - t0, "pt": -1, "ct": -1,
                            "err": str(e)[:80]})

def run(C, DUR):
    stop = threading.Event()
    results = []
    threads = [threading.Thread(target=worker, args=(stop, results)) for _ in range(C)]
    for t in threads: t.start()
    time.sleep(DUR)
    stop.set()
    for t in threads: t.join()
    return results

def summarize(name, results, elapsed, out):
    ok = [r for r in results if r["ct"] > 0]
    err = [r for r in results if r["ct"] <= 0]
    tot = sum(r["ct"] for r in ok)
    dts = sorted(r["dt"] for r in ok)
    per = sorted(r["ct"] / r["dt"] for r in ok if r["dt"] > 0)
    s = (f"{name}: requests={len(ok)} errors={len(err)} wall={elapsed:.1f}s "
         f"aggregate={tot/elapsed:.1f} tok/s | per-request p50={per[len(per)//2]:.1f} "
         f"p95={per[int(len(per)*0.95)]:.1f} tok/s" if per else f"{name}: no successful requests")
    print(s)
    with open(out, "a") as f:
        f.write(json.dumps({"test": name, "elapsed": elapsed, "requests": len(ok),
                            "errors": len(err), "total_completion_tokens": tot,
                            "aggregate_tok_s": tot / elapsed,
                            "per_request_p50": per[len(per)//2] if per else None,
                            "per_request_p95": per[int(len(per)*0.95)] if per else None,
                            "results": results}) + "\n")
    return tot / elapsed

if __name__ == "__main__":
    concurrency = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    name = sys.argv[3] if len(sys.argv) > 3 else f"c{concurrency}"
    out = sys.argv[4] if len(sys.argv) > 4 else "ceiling.jsonl"
    t0 = time.time()
    res = run(concurrency, duration)
    summarize(name, res, time.time() - t0, out)
