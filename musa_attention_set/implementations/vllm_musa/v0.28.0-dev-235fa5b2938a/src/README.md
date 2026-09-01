<p align="center">
  <img src="assets/logo.png" alt="vLLM-MUSA" width="60%">
</p>

<h2 align="center">High-performance LLM serving on Moore Threads MUSA</h2>

<p align="center">
  <a href="README_CN.md">中文</a> ·
  <a href="https://github.com/MooreThreads/vllm-musa/issues">Issues</a> ·
  <a href="docs/cookbook/README.md">Serving recipes</a> ·
  <a href="docs/installation.md">Documentation</a>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10-blue.svg" alt="Python 3.10"></a>
</p>

---

*New model support images* 🔥

- [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:qwen38-flash-next`
- [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:glm53-flash`
- [Hy4-preview](https://huggingface.co/tencent/Hy4-preview) — `registry.mthreads.com/mcconline/inference/vllm/vllm-openai:hy4-preview`

These model-specific tags are maintained independently of this branch; verify
the exact image and model revision before production use.

---

## About

vLLM-MUSA is the Moore Threads backend for [vLLM](https://github.com/vllm-project/vllm),
providing an OpenAI-compatible inference and serving engine for MUSA GPUs.

- v0.28.0-dev development line with the vLLM V1 engine.
- Uses the pinned MUSA stack: PyTorch/torch_musa 2.11.0.post1 (MUSA 5.2.0),
  MATE 0.2.6, and torchada 0.1.83; this branch remains an upgrade candidate.
- MUSA-native attention, communication, custom ops, and compilation support.
- Per-checkpoint recipes for Qwen and DeepSeek-V4-Flash on S5000.

## Getting started

### Install

Use the published v0.28.0 release image:

```bash
export VLLM_MUSA_IMAGE=registry.mthreads.com/mcconline/inference/vllm/vllm-openai:v0.28.0
docker pull "${VLLM_MUSA_IMAGE}"
```

The image is built independently of this branch. Verify its installed package
versions against `third_party/PINS`; build the local image from the
[installation guide](docs/installation.md) when exact branch parity is needed.

- [Installation and source build](docs/installation.md)
- [Docker build guide](docker/README.md)
- [Configuration reference](docs/configuration.md)

### Quickstart

```bash
vllm serve /path/to/model \
  --trust-remote-code \
  --served-model-name my-model
```

The command above assumes that `vllm` is installed and on the current
environment's `PATH`. For the release image, use the container launch shape
in the [Docker guide](docker/README.md).

The OpenAI-compatible endpoint is available at `http://localhost:8000/v1`.
For recommended tensor-parallel, scheduler, and speculative-decoding settings,
use the [serving cookbook](docs/cookbook/README.md).

## Documentation

- [Serving cookbook](docs/cookbook/README.md)
- [Installation](docs/installation.md)
- [Python and OpenAI-compatible usage](docs/usage.md)
- [Configuration](docs/configuration.md)
- [Development](docs/development.md)
- [MUSA developer guide](docs/mdm-developer-guide.md)

## Supported release

| vLLM-MUSA | PyTorch/MUSA | Engine | Status |
|---|---|---|---|
| v0.28.0-dev | 2.11.x | V1 engine; Runner V2 where supported | Upgrade candidate |

The V1 engine may automatically select Model Runner V2 for supported model
architectures; both runners are supported on MUSA.
This branch tracks the upstream v0.28.0 commit in `third_party/PINS` and
intentionally keeps the MUSA PyTorch 2.11.x stack; promote it to a supported
release only after build and model-smoke validation.

## Contributing

See the [development guide](docs/development.md) for editable installs, tests,
patch workflow, and issue reports. Contributions and bug reports are welcome
through [GitHub Issues](https://github.com/MooreThreads/vllm-musa/issues).

## Related projects

- [Moore Threads](https://www.mthreads.com/)
- [vLLM](https://github.com/vllm-project/vllm)
- [vLLM documentation](https://docs.vllm.ai/en/latest/)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [torchada](https://github.com/MooreThreads/torchada)
- [torch_musa](https://github.com/MooreThreads/torch_musa)
- [MATE](https://github.com/MooreThreads/mate)
- [mthreads-ml-py](https://github.com/MooreThreads/mthreads-ml-py)

## License

vLLM-MUSA is released under the [Apache License 2.0](LICENSE).
