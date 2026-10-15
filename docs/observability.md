# Observability stack (optional, not required for inference)

The cluster runs a small monitoring/alerting layer on the **head** node:

| Piece | Unit / file | What it does |
|---|---|---|
| `node-exporter` | `systemd/node-exporter.service` | host metrics (CPU/mem/disk/network) for Prometheus |
| `nvidia-gpu-exporter` | `systemd/nvidia-gpu-exporter.service` | GPU metrics (utilization, memory, power, thermals) |
| Spark vLLM engine health | `spark-vllm-health.service` / `spark-vllm-watchdog` | liveness probe + auto-restart (see README; this is the supervised part) |
| Grafana dashboards | `dashboards/dgx-dashboard-shot.mjs`, `dashboards/dgx-dashboard-with-thermal-panels.json`, `dashboards/grafana-dashboard-dgx*.png` | pre-built views (GPU + thermal panels); JSON is the importable dashboard |
| Power alerting | `dashboards/ha-dgx-power-alert.yaml` | Home Assistant automation: Pushover when DGX smart plugs are OFF ≥5 min (µ once/4 h, 08:00–20:00 only; alert-only, no auto power-on) |

Prometheus, Grafana, and Home Assistant themselves are **not** packaged here —
they are external services assumed to be running (scrape the exporters from
anywhere; the units are user-level and need no root).

Deploy: `scripts/deploy.sh` installs the units; enable with
`systemctl --user enable --now node-exporter nvidia-gpu-exporter`

> The inference stack (units `spark-vllm-*`) is independent of this layer —
> monitoring can be added or removed without touching model serving.
