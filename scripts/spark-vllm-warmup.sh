#!/bin/bash
# Warm up DeepSeek-V4-Flash-Vision-Exp vision/MM kernels (Triton JIT) at startup,
# before user traffic, so compile spikes don't land mid-session.
set -u
HEALTH=http://127.0.0.1:8000/health
CHAT=http://127.0.0.1:8000/v1/chat/completions
MODEL=deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
for i in $(seq 1 60); do
  curl -sf -m 5 "$HEALTH" >/dev/null && break
  sleep 10
done
IMG='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
curl -sf -m 300 -X POST "$CHAT" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"$IMG\"}},{\"type\":\"text\",\"text\":\"Describe this image in one word.\"}]}],\"max_tokens\":8,\"chat_template_kwargs\":{\"thinking\":false}}" \
  >/tmp/spark-vllm-warmup.out 2>&1 && echo "WARMUP OK: $(python3 -c 'import json;d=json.load(open("/tmp/spark-vllm-warmup.out"));print(d["usage"])' 2>/dev/null || echo done)" || { echo "WARMUP FAILED"; head -c 300 /tmp/spark-vllm-warmup.out; }
exit 0
