# Fresh install — 2-node DGX Spark dual stack (from scratch)

Reproduce the cluster exactly as configured here. Assumes two GB10 boxes
(DGX Spark / ASUS Ascent GX10), 128 GB unified memory each, DGX OS.

## 0. Node layout

| | head | worker |
|---|---|---|
| hostname alias | gx10-01 | gx10-02 |
| CX7 mesh IPs | 192.168.177.11 (+ .178.11 second ring) | 192.168.177.12 (+ .178.12) |
| ConnectX-7 ports | enp1s0f1np1 (RDMA rocep1s0f1), enP2p1s0f1np1 (second) | same naming |
| user | kambiz (rootless podman + systemd user units) | kambiz |

Change `HEAD_IP`/`WORKER_IP`/`IFACE`/`IB_HCA` below to match your fabric.

## 1. Networking (both ports → 400 Gb/s aggregate)

Two ConnectX-7 200 GbE ports per node. Cable **both** ports port-to-port
(no switch needed for 2 nodes):

```
head QSFP-A <-> worker QSFP-A     (ring 1: enp1s0f1np1 / 192.168.177.x)
head QSFP-B <-> worker QSFP-B     (ring 2: enP2p1s0f1np1 / 192.168.178.x)
```

Each physical port exposes **two** Linux interfaces (2× PCIe Gen5 x4 into the
GB10 SoC), so with both cables you get four logical links; NCCL/RoCE uses them
all. Assign IPs to every interface used (netplan on DGX OS):

```yaml
network:
  version: 2
  ethernets:
    enp1s0f1np1:
      addresses: [192.168.177.11/24]     # head / .12 on worker
    enP2p1s0f1np1:
      addresses: [192.168.178.11/24]     # head / .12 on worker
```

Verify: `ip -4 addr`, `ls /sys/class/infiniband/` (expect `rocep1s0f1` etc.),
`ping 192.168.177.12`.

## 2. Software baseline

```bash
sudo dnf install -y podman git python3 sshpass   # (or your package manager)
# ssh keys both nodes: ssh-keygen; ssh-copy-id gx10-02
mkdir -p ~/.local/bin
```

Check podman GPU + RDMA support as your user:

```bash
podman info | grep -iE "cdi|nvidia"   # GPU device passthrough
podman run --rm --device nvidia.com/gpu=all nvidia/cuda:12.6.0-base nvidia-smi
```

The b12x launcher uses `--gpus all`; podman accepts it on this kit.

## 3. Systemd user units (head)

```bash
cp systemd/spark-vllm-podman.service ~/.config/systemd/user/
cp systemd/spark-vllm-health.service  ~/.config/systemd/user/
# drop-in that swaps ExecStart to the model dispatcher:
mkdir -p ~/.config/systemd/user/spark-vllm-podman.service.d
cp systemd/override.conf ~/.config/systemd/user/spark-vllm-podman.service.d/
systemctl --user daemon-reload
systemctl --user enable --now spark-vllm-health.service
```

`override.conf`:

```
[Service]
ExecStart=
ExecStart=/home/kambiz/.local/bin/spark-vllm-podman-start-select
```

## 4. Install scripts

```bash
./scripts/deploy.sh            # installs scripts + config on this node (head)
./scripts/deploy.sh --all      # also push runbook/image step to worker (see below)
```

`deploy.sh` copies `scripts/` to `~/.local/bin`, `config/` to
`~/.config/spark-vllm/` and re-reads model.conf from the template.

## 5. Weights (HF token)

```bash
huggingface-cli login                      # or export HF_TOKEN=hf_xxx
# DeepSeek (b12x):
hf download --token "$HF_TOKEN" deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --local-dir ~/.cache/huggingface/hub/...
# Qwen3.8-Flash-Next (native stack):
hf download --token "$HF_TOKEN" nvidia/Qwen3.8-Flash-Next-NVFP4 --local-dir ~/.cache/huggingface/hub/...
# copy to worker (use -L or tar -ch: rsync -a alone only copies the HF cache
# symlinks, NOT the blob bytes — verified pitfall on this kit):
rsync -aL ~/.cache/huggingface/ gx10-02:~/.cache/huggingface/
# (or, streaming over the CX7 mesh using real bytes, keeping paths intact:)
# tar -chf - ~/.cache/huggingface | ssh 192.168.177.12 "tar -C ~ -xf -"
```

## 6. DeepSeek/b12x stack

```bash
# Our fork (podman-patched, CONTAINER_RT) — upstream: eugr/spark-vllm-docker
git clone https://github.com/kambiz-aghaiepour/spark-vllm-docker.git ~/spark-vllm-docker
cd ~/spark-vllm-docker && ./build-and-copy.sh       # builds MXFP4 image, copies to worker
# model config already present via deploy.sh (flags/deepseek-v4.conf)
./scripts/switch.sh deepseek-ai/DeepSeek-V4-Flash-Vision-Exp --apply
```

Image build note: `Dockerfile.mxfp4` builds from `nvcr.io/nvidia/pytorch`
(NGC login required for `podman pull`), pins `christopherowen/vllm@045293d`
(mxfp4 lineage), and applies the fork's launch-time patch overlays
(`mods/`), including the `mods/drop-caches` page-cache helper and the applied
vLLM PR patches (`local-inference-lab/vllm#669` by logprobz; NCCL load-order
fix tracked in `vllm-project/vllm#42354`). Podman launch support comes from
upstream PR **eugr/spark-vllm-docker#131** (*"Add podman support via
CONTAINER_RT"*, **Sebastian Jug / sjug**, commit `55cb129`).

## 7. Qwen3.8-Flash-Next native stack

```bash
./scripts/podman-adapt-runbook.sh     # clones MiaAI-Lab/... and podman-adapts it
./scripts/switch.sh nvidia/Qwen3.8-Flash-Next-NVFP4 --apply
```

`podman-adapt-runbook.sh` clones
`https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` to
`~/miaai-qwen38`, converts `docker`→`podman` in `start.sh`/`stop.sh`, and
writes the fabric `.env` (port 8000, CX7 IPs, IFACE/IB_HCA above). Image is
pulled once on the head and synced to the worker by the runbook.

## 8. Verify

```bash
vllm-model-select --status      # model.conf / service / watchdog / live engine
curl -s http://127.0.0.1:8000/v1/models
```

## Expected timings (this kit)

- DeepSeek load + ready: ~7.5 min
- flash-next (native) load + ready: ~10–15 min (first run includes image pull)
- Switch back and forth: 7–13 min each
