# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KernelBench is a benchmark for evaluating LLMs' ability to generate efficient GPU kernels. It tests whether models can transpile PyTorch operators into custom CUDA/Triton/CuTe/TileLang kernels that are both correct and performant.

The benchmark has 4 levels:
- **Level 1**: Single-kernel operators (100 problems) - matmul, conv, layer norm
- **Level 2**: Simple fusion patterns (100 problems) - conv+bias+relu, matmul+scale+sigmoid
- **Level 3**: Full model architectures (50 problems) - MobileNet, VGG, MiniGPT, Mamba
- **Level 4**: HuggingFace model architectures

## Setup

### Using uv (recommended)
```bash
uv sync
```

### Using pip
```bash
conda create --name kernel-bench python=3.10
conda activate kernel-bench
pip install -e .
```

Configure API keys by copying `.env.example` to `.env` and filling in keys for LLM providers (OpenAI, Anthropic, Google, DeepSeek, Together AI, etc.).

## Common Commands

### Single Problem Generation + Evaluation
```bash
python3 scripts/generate_and_eval_single_sample.py dataset_src=huggingface level=1 problem_id=1
```

### Batch Generation
```bash
python3 scripts/generate_samples.py run_name=my_run dataset_src=huggingface level=1 num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0
```

### Batch Evaluation
```bash
python3 scripts/eval_from_generations.py run_name=my_run dataset_src=local level=1 num_gpu_devices=8 timeout=300
```

### Compute Benchmark Metrics
```bash
python3 scripts/benchmark_eval_analysis.py run_name=my_run level=1 hardware=L40S baseline=baseline_time_torch
```

### Generate Baseline Times (for new hardware)
```bash
python3 scripts/generate_baseline_time.py
python3 scripts/generate_baseline_time_modal.py  # for Modal cloud
```

### Quick Check Single Kernel
```bash
python3 scripts/run_and_check.py
```

### Run Tests
```bash
pytest src/unit_tests/
```

## Key Configuration Parameters

Scripts use `pydra` for configuration (CLI args override defaults):

- `dataset_src`: "huggingface" or "local"
- `level`: 1-4 (benchmark level)
- `problem_id`: problem number within level
- `gpu_arch`: GPU architecture - "Ada", "Hopper", "Ampere", "Turing", or SM version
- `backend`: "cuda", "triton", "cute", "tilelang"
- `precision`: "fp32", "fp16", "bf16"
- `server_type`: LLM provider - "openai", "anthropic", "google", "deepseek", "together", "local", etc.
- `model_name`: model identifier (e.g., "gpt-4", "claude-3-opus", "deepseek-chat")
- `eval_mode`: "local" (requires GPU) or "modal" (cloud GPU)
- `timing_method`: "cuda_event", "do_bench", "do_bench_impl", "host_time"
- `prompt_option`: "zero_shot", "one_shot", "few_shot"

## Architecture

### Core Library (`src/`)
- `eval.py` - Correctness checking and timing evaluation
- `dataset.py` - Dataset loading (local files or HuggingFace)
- `utils.py` - LLM querying via LiteLLM, code extraction, GPU arch setting
- `compile.py` - CUDA kernel compilation and caching
- `timing.py` - Multiple timing methods (CUDA events, Triton do_bench, host timing)
- `score.py` - Metric calculation including `fast_p` (fraction correct AND faster than threshold)
- `prompt_constructor_toml.py` - TOML-based prompt composition system

### Scripts (`scripts/`)
- `generate_and_eval_single_sample.py` / `*_modal.py` - Single problem workflow
- `generate_samples.py` - Batch LLM generation
- `eval_from_generations.py` - Evaluate pre-generated kernels
- `benchmark_eval_analysis.py` - Compute benchmark metrics

### Benchmark Dataset (`KernelBench/`)
Contains problem files organized by level. Each problem is a Python file with a `Model` class and `get_inputs()`/`get_init_inputs()` functions.

### Prompts (`src/prompts/`)
- `prompts.toml` - TOML configuration defining prompt templates, backends, precision modes
- Example files (`model_*.py`) - Few-shot examples for different backends
- `hardware/gpu_specs.py` - GPU hardware specifications for prompts

### Results (`results/timing/`)
Pre-computed PyTorch baseline times for various GPUs (H100, L40S, A100, T4, etc.) and configurations (eager, torch.compile variants).

## Evaluation Flow

1. Load problem from dataset (local or HuggingFace)
2. Construct prompt using TOML templates + optional hardware info
3. Query LLM via LiteLLM
4. Extract code from response
5. Compile kernel with specified GPU architecture
6. Check correctness (n_correctness times with random inputs)
7. Measure timing (reference PyTorch vs generated kernel)
8. Compute metrics: correctness rate, speedup, `fast_p` score

## Key Metric: `fast_p`

The `fast_p` metric measures fraction of tasks that are both correct AND have speedup > p:
- `fast_0` = correctness rate
- `fast_1` = fraction correct and faster than PyTorch
- `fast_2` = fraction correct and at least 2x faster
