# NOTICE

This repository documents and packages the orchestration for a two-node
NVIDIA DGX Spark (GB10) cluster running two inference stacks.

## Attribution

The following are **upstream projects, not vendored here** — they are cloned /
pulled at runtime by the scripts and referenced in the README:

| Upstream | What it is | License | Where it lives at runtime |
|---|---|---|---|
| `kambiz-aghaiepour/spark-vllm-docker` | **our fork** of the DeepSeek/b12x launcher; podman support via `CONTAINER_RT` (commit `55cb129`) + applied vLLM PR patches (`local-inference-lab/vllm#669`, `vllm-project/vllm#42354`) | see upstream | `~/spark-vllm-docker` |
| `eugr/spark-vllm-docker` | **upstream** of the above fork (DeepSeek/b12x cluster launcher + MXFP4 image build) | see upstream | upstream repository |
| `christopherowen/spark-vllm-mxfp4-docker` | image/Dockerfile lineage (vLLM `045293d`, FlashInfer, CUTLASS forks) | see upstream | upstream reference only |
| `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` | Qwen3.8-Flash-Next runbook + `vllm/vllm-openai:qwen38-flash-next` image + overlay patches | **AGPL-3.0** | `~/miaai-qwen38` (cloned by `podman-adapt-runbook.sh`) |
| NVIDIA docs (`docs.nvidia.com/dgx/dgx-spark`, `build.nvidia.com/spark`) | DGX Spark clustering / Connect-Two-Sparks | NVIDIA | n/a (docs) |

## License of this repository

AGPL-3.0. The MiaAI-side launch integration (`miaai_launch.sh` and the
`docker`→`podman` adaptation script) derives from the AGPL-3.0 MiaAI runbook;
the remainder is original work of the author. This is used internally; before
any external redistribution, satisfy the upstream AGPL terms.

## Security note

The repository is deliberately free of credentials: no HF tokens, no cloud
API keys. Tokens are provided at runtime via `HF_TOKEN`/`huggingface-cli login`
and (for NVIDIA NGC image pulls) an NGC login. Review `scripts/` before running
on a new host; the launchers render transient scripts with mode 0600.
