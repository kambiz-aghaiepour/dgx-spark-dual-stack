# NOTICE

This repository documents and packages the orchestration for a two-node
NVIDIA DGX Spark (GB10) cluster running two inference stacks.

## Attribution

The following are **upstream projects, not vendored here** — they are cloned /
pulled at runtime by the scripts and referenced in the README:

| Upstream | What it is | License | Where it lives at runtime |
|---|---|---|---|
| `kambiz-aghaiepour/spark-vllm-docker` | **our fork** of the DeepSeek/b12x launcher; podman support via `CONTAINER_RT` (commit `55cb129`) + applied vLLM PR patches (`local-inference-lab/vllm#669`, `vllm-project/vllm#42354`) | see upstream | `~/spark-vllm-docker` |
| `eugr/spark-vllm-docker` | **upstream** of the above fork (DeepSeek/b12x cluster launcher + MXFP4 image build). Podman support: **PR #131 "Add podman support via CONTAINER_RT" by Sebastian Jug ([sjug](https://github.com/sjug)**, commit `55cb129`) | see upstream | upstream repository |
| `christopherowen/spark-vllm-mxfp4-docker` | image/Dockerfile lineage (vLLM `045293d`, FlashInfer, CUTLASS forks) | see upstream | upstream reference only |
| `MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks` | Qwen3.8-Flash-Next runbook + `vllm/vllm-openai:qwen38-flash-next` image + overlay patches | **AGPL-3.0** | `~/miaai-qwen38` (cloned by `podman-adapt-runbook.sh`) |
| `MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` | **upstream** of the current default stack: the two-Spark DeepSeek-V4.1-Flash-1017 recipe (native-Zig TensorFold server, image build, weight fetch, NFS, unit) | **Apache-2.0** | upstream repository (`~/mia-dsv41` is our fork, below) |
| `kambiz-aghaiepour/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks` | **our fork** of the above, branch `podman-support`: adds the container-runtime switch (`CONTAINER_RT`, `scripts/runtime.sh`) so the pair runs under rootful podman; docker stays the recipe default. Offered upstream as PR [#113](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/113) | **Apache-2.0** (unchanged) | `~/mia-dsv41` (cloned + pinned by `scripts/setup-mia-recipe.sh`) |
| `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` | **upstream** of the `glm53` profile: the two-Spark GLM-5.3-Flash-EXL3 recipe (image build, gated Ablit weight fetch, DFlash2 drafter, CUDA extension) | see upstream | upstream repository (`~/glm53` is our fork, below) |
| `kambiz-aghaiepour/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold` | **our fork** of the above, branch `podman-support`: the same container-runtime switch, plus the rootful-podman fixes (image-store reads, `HF_TOKEN` by value, kernel-cache mount source, no `--shm-size` beside `--ipc=host`); docker stays the recipe default. Offered upstream as PR [#246](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/246) | see upstream (unchanged) | `~/glm53` (cloned + pinned by `scripts/setup-glm-recipe.sh`) |
| NVIDIA docs (`docs.nvidia.com/dgx/dgx-spark`, `build.nvidia.com/spark`) | DGX Spark clustering / Connect-Two-Sparks | NVIDIA | n/a (docs) |

## License of this repository

AGPL-3.0. The MiaAI-side launch integration (`miaai_launch.sh` and the
`docker`→`podman` adaptation script) derives from the AGPL-3.0 MiaAI runbook,
and the container-runtime switch in `podman-support` modifies the
**Apache-2.0** DeepSeek-V4.1-Flash-1017 recipe — that fork keeps its Apache-2.0
`LICENSE` and notices, and is offered upstream as PR #113. The same switch in the
**GLM-5.3-Flash-EXL3** fork (`~/glm53`, PR #246) keeps *its* upstream licence and
notices. All three MiaAI projects stay in their own clones at runtime, not vendored
here. The remainder is original work of the author. This is used internally; before
any external redistribution, satisfy the upstream terms (AGPL-3.0 for the Qwen
runbook, and each TensorFold recipe's own licence for the DeepSeek and GLM forks).

## Security note

The repository is deliberately free of credentials: no HF tokens, no cloud
API keys. Tokens are provided at runtime via `HF_TOKEN`/`huggingface-cli login`
and (for NVIDIA NGC image pulls) an NGC login. Review `scripts/` before running
on a new host; the launchers render transient scripts with mode 0600.
