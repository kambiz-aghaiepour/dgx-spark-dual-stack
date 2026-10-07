#!/usr/bin/env python3
"""Throwaway concurrency/type sweep for the flash-next native stack (port 8000).
Measures aggregate tok/s, per-stream p50/p95, TTFT for:
  - decode-heavy chat (C sweep)
  - long context (16K/64K in -> short out)
  - vision (generated PNG, base64 data URL)
  - reasoning-heavy chat
Placeholder-free; results printed as JSON lines.
"""
import argparse, base64, json, struct, threading, time, urllib.request, zlib

def make_png(w=512, h=512, rgb=(48, 63, 159)):
    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))

def post(url, body, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    return r, dt

def worker(url, cfg, nreq, res):
    for _ in range(nreq):
        body = {"model": cfg["model"], "max_tokens": cfg["max_tokens"],
                "temperature": 0.0,
                "messages": [{"role": "user", "content": cfg["content"]}]}
        if cfg.get("image"):  # content list with image + text
            body["messages"][0]["content"] = [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{cfg['image']}"}},
                {"type": "text", "text": "Describe this image in a few sentences."}]
        try:
            r, dt = post(url, body)
            u = r["usage"]
            res.append({"pt": u["prompt_tokens"], "ct": u["completion_tokens"],
                        "dt": dt})
        except Exception as e:
            res.append({"pt": -1, "ct": -1, "err": str(e)[:100]})

def run(url, cfg, C, duration):
    res, stop = [], threading.Event()
    def once():
        while not stop.is_set():
            worker(url, cfg, 1, res)
    # one-shot workers: issue requests for `duration`, then stop starting new
    threads = [threading.Thread(target=once) for _ in range(C)]
    for t in threads: t.start()
    time.sleep(duration); stop.set()
    for t in threads: t.join()
    return res

def report(name, res, duration):
    ok = [r for r in res if r["ct"] > 0]
    agg = sum(r["ct"] for r in ok) / duration
    per = sorted(r["ct"] / r["dt"] for r in ok if r["dt"] > 0)
    p50 = per[len(per)//2] if per else 0
    p95 = per[int(len(per)*0.95)] if per else 0
    ttft_est = "n/a"
    print(json.dumps({"test": name, "ok": len(ok), "err": len(res) - len(ok),
                      "aggregate_tok_s": round(agg, 1),
                      "p50_tok_s": round(p50, 1), "p95_tok_s": round(p95, 1),
                      "wall_s": duration}))
    return agg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.1.39:8000/v1/chat/completions")
    ap.add_argument("--model", default="nvidia/Qwen3.8-Flash-Next-NVFP4")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--name", default="sweep")
    ap.add_argument("--kind", default="chat", choices=["chat", "long16k", "long64k",
                                                       "vision", "reason"])
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()

    n = 1550
    if a.kind == "chat":
        content = ("Write the numbers from 1 to 1500, one per line, in order "
                   "without stopping."); a.max_tokens = 1024
    elif a.kind == "long16k":
        content = ("Prefix: " + ("the quick brown fox jumps over the lazy dog " * 900)
                   + "\n\nSummarize the above in 3 sentences."); a.max_tokens = 256
    elif a.kind == "long64k":
        content = ("P: " + ("lorem ipsum dolor sit amet consectetur adipiscing "
                            "elit sed do eiusmod tempor " * 1900)
                   + "\n\nSummarize in 2 sentences."); a.max_tokens = 128
    elif a.kind == "vision":
        content = ""  # replaced with image
    else:  # reason
        content = ("How many letter 'r' are in the word 'strawberry'? "
                   "Show your reasoning before the final answer.")
        a.max_tokens = 512

    cfg = {"model": a.model, "max_tokens": a.max_tokens,
           "content": content, "image": base64.b64encode(make_png()).decode()
           if a.kind == "vision" else None}
    report(a.name, run(a.url, cfg, a.concurrency, a.duration), a.duration)
