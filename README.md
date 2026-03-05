# KernelBench: Can LLMs Write Efficient GPU Kernels? [ICML '25]
A benchmark and environment for evaluating LLMs' ability to generate efficient GPU kernels

Specifically we task LLM to generate correct and efficient CUDA / DSL kernels for PyTorch programs on a target GPU.

[arXiv](https://arxiv.org/html/2502.10517v1) | [blog post](https://scalingintelligence.stanford.edu/blogs/kernelbench/) | [HuggingFace Dataset](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) 

<img src="./assets/figures/KernelBenchMascot.png" width="200">

## Versions
The latest stable version will be on `main` branch. We continue to update and improve the repo. 
- [v0.1](https://github.com/ScalingIntelligence/KernelBench/tree/v0.1) - See [blog](https://scalingintelligence.stanford.edu/blogs/kernelbenchv01/)
- [v0](https://github.com/ScalingIntelligence/KernelBench/tree/v0) - Original Release


The Huggingface [dataset](https://huggingface.co/datasets/ScalingIntelligence/KernelBench) is updated to v0.1.

This repo provides core functionality for KernelBench and an easy-to-use set of scripts for evaluation. It is not intended to provide complex agentic scaffolds that solve this task; we recommend cloning and modifying this repo for your experiment, or using it as a git submodule.

## 👋 Task Description
We structure the problem for LLMs to transpile operators described in PyTorch to CUDA kernels, at whatever level of granularity they desire.
![KernelBenchMascot](./assets/figures/KernelBenchWorkFlow.png)

We construct KernelBench to have 4 Levels of categories:
- **Level 1 🧱**:  Single-kernel operators (100 Problems)
    The foundational building blocks of neural nets (Convolutions, Matrix multiplies, Layer normalization)
- **Level 2 🔗**:  Simple fusion patterns (100 Problems)
    A fused kernel would be faster than separated kernels (Conv + Bias + ReLU, Matmul + Scale + Sigmoid)
- **Level 3 ⚛️**:  Full model architectures (50 Problems)
    Optimize entire model architectures end-to-end (MobileNet, VGG, MiniGPT, Mamba) 
- **Level 4 🤗**:  Level Hugging Face 
    Optimize whole model architectures from HuggingFace

We are actively extending KernelBench to other DSLs beyond `cuda` as well (see below), as well as AMD GPU support.

## ⚖️ Evaluation
#### Methodology
To evaluate model-generated kernels, we need to check if they:
- **are correct ✅**: check against reference torch operators `n_correctness` times on randomized inputs.
- **are performant ⏱️**: compare against reference torch operators `n_trial` times to measure speedup between runtimes.

Check out `src/eval.py` for details on how we implement correctness check and timing and `EVAL.md` for notes on evaluation and benchmarking guidelines [WIP].

We provide a convenient script `scripts/run_and_check.py` to evaluate one single sample source code against a reference source code, check correctness and compute speedup. You can use this to evaluate a kernel either locally or remotely by setting `eval_mode=local` or `eval_mode=modal`.

#### Overall Benchmark Metric

Since we need to capture **both** correctness and performance, we define a metric `fast_p`: fraction of tasks that are both correct and have a speedup greater than threshold `p`; speedup is computed as the ratio of PyTorch reference wall-clock time to generated kernel time.

Some examples to illustrate this metric that filters based on speedups:
* `fast_1` is the fraction of tasks that LM-generated kernels are both correct and **faster** than PyTorch baseline
* `fast_2` is the fraction of tasks that LM-generated kernels are both correct and **at least 2x faster** than PyTorch baseline
* `fast_0` is the fraction of tasks that LM-generated kernels are **correct**. (same as correctness rate)

You can increase speedup threshold `p` to make the task more challenging.


#### Compute Overall Benchmark Performance

We provide a script `scripts/greedy_analysis.py` to compute the overall benchmark performance. 
Since we need to capture **both** correctness and performance, we use a metric `fast_p`: fraction of tasks that are both correct and have a speedup greater than threshold `p`; speedup is computed as the ratio of PyTorch reference wall-clock time to generated kernel time.

<!-- TODO: update to provide fast_p measurement script -->

## 🔍 Directory Structure
We organize the repo into the following structure:
```
KernelBench/
├── assets/
├── KernelBench/ # Benchmark dataset files
├── src/kernelbench/ # KernelBench logic code
│   ├── unit_tests/  
│   ├── prompts/
│   ├── ....
├── scripts/ # helpful scripts to run the benchmark
├── results/ # baseline times across hardware 
├── runs/ # where your runs will be stored
├── notebooks/ # example notebooks for analysis
├── pyproject.toml # Project configuration and dependencies
```

## 🔧 Set up

We have transitioned to using `pyproject.toml` and `uv` for dependency management. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already

```bash
# Install base dependencies (works without a local GPU)
uv sync

# Install with AMD ROCm backend (ROCm>=7.1 is required)
uv add torch --index pytorch=https://download.pytorch.org/whl/rocm7.1

# Install with GPU dependencies (for local GPU evaluation)
uv sync --extra gpu

# Run commands with uv (which invoke the right env)
uv run python scripts/<script_name>.py ...
```

For AMD GPU aka ROCm backend (ROCm>=7.1), please add `uv remove torch && uv add torch --index pytorch=https://download.pytorch.org/whl/rocm7.1` for RoCm compatible PyTorch to configure your dependencies. Running in a docker image is recommended for this due to complexity of ROCm setup.

You can still use `conda (python=3.10)` to create your environment and install dependencies with `requirements.txt`.

We use `litellm` for API calls. Please set your keys by creating a `.env` following our `.env.example`.

Running and profiling kernels require a GPU.
If you don't have a GPU available locally, you can set up [Modal](https://modal.com/) for cloud serverless GPU evaluation. Set up your modal token after creating an account by running `modal token new`. Then, use the `generate_and_eval_single_sample_modal.py` script.

You can also try out our [tutorial notebook](https://bit.ly/kernelbench-neurips-colab) (also in notebooks/tutorial.ipynb) with Google Colab.

## 🚀 Usage
### Run on a single problem 
It is easier to get started with a single problem. This will fetch the problem, generate a sample, and evaluate the sample. 

```bash
# for example, run level 2 problem 40 from huggingface and use google gemini 2.5 flash for generation

uv run python scripts/generate_and_eval_single_sample.py dataset_src=huggingface level=2 problem_id=40 server_type=google model_name=gemini/gemini-2.5-flash

# dataset_src could be "local" or "huggingface"
# add .verbose_logging for more visibility
```

**What you might need to modify**
* **`gpu_arch`** - Depending on your GPU, you might need to adjust the `gpu_arch` argument to reflect your hardware.
* **`precision`** - You can specify the precision of tensor by `precision=fp32`. Currently all of our reported results are `fp32` but we added support for `fp16` & `bf16`.
*  **`backend`** - We are also supporting other GPU programming languages beyond `cuda`. For example, simply specify `backend=triton` or `backend=hip`. For now we support NVIDIA GPUs with programming frameworks and DSLs: `cuda`, `triton`, `cute`, `tilelang`, `thunderkittens`. 

Note for AMD GPUs: Use `hip` backend, `gpu_arch` currently supported: `gfx942`, `gfx950`.

Note on setting up ThunderKittens (TK) locally: to use `backend=thunderkittens`, you need to git clone the ThunderKittens repo and set the following environment variable to point to your local ThunderKittens directory, `export THUNDERKITTENS_ROOT=<PATH to ThunderKittens folder>`, and all ThunderKitten programs as shown in the [example](src/kernelbench/prompts/model_new_ex_add_thunderkittens.py), should contain `tk_root = os.environ.get("THUNDERKITTENS_ROOT", "/root/ThunderKittens")`, which enable the kernel to include the right TK primitives. In addition, we only support BF16 for TK right now.

Check the config fields for a comprehensive set of options. Note we provide the model with a one-shot example by default along with the minimum set of info; you can check out other prompt settings or construct your own in `src/prompt_constructor_toml.py`.

### Run on all problems 

```bash
# 1. Generate responses and store kernels locally to runs/{run_name} directory
uv run python scripts/generate_samples.py run_name=test_hf_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0

# 2. Evaluate on all generated kernels in runs/{run_name} directory
uv run python scripts/eval_from_generations.py run_name=test_hf_level_1 dataset_src=local level=1 num_gpu_devices=8 timeout=300

# If you like to speedup evaluation, you can use parallelize compilation on CPUs before getting to evaluation on GPUs
# add build_cache=True and num_cpu_workers=<num_cpu_workers> to the command
```
### Analyze the eval results to compute Benchmark Performance
We provide `scripts/benchmark_eval_analysis.py` to analyze the eval results to compute success rate, timing metric, and overall benchmark performance  `fast_p`. 

```bash
uv run python scripts/benchmark_eval_analysis.py run_name=test_hf_level_1 level=1 hardware=L40S_matx3 baseline=baseline_time_torch
```
If you are using a different hardware, you can generate the baseline time with `scripts/generate_baseline_time.py` script.
We provide some reference baseline times for a variety of NVIDIA GPUs across generations in `results/timing`, but we recommend you generate your own baseline time for more accurate results (cluster power, software version, all affect timing results). See `results/timing/README.md` for more details.

## 🛣️ Upcoming Roadmap
Check out our [roadmap](https://github.com/ScalingIntelligence/KernelBench/issues/74) for what we plan to add as features. We welcome community contributions and discussions in these directions. 

## 🔌 Integration 
You can also use KernelBench as a library for your projects, for example: `from kernelbench import timing`, `from kernelbench import eval as kb_eval`, or `from kernelbench.utils import set_gpu_arch`.

- **Adapter with Harbor** — We are integrating with [Harbor](https://harborframework.com/docs) to enable higher-throughput eval and richer evaluation of agentic performance beyond model pass@1/k. *([Ongoing](https://github.com/harbor-framework/harbor/pull/999))*

- **Multi-Turn / Test-Time Scaling** — [Caesar](https://github.com/ScalingIntelligence/caesar) is our throughput-oriented multi-turn inference engine (ICML '25), used for the iterative refinement experiments in the paper. It runs generation trajectories in batch, feeding back correctness, runtime, and profiling signals across turns for sequential test-time scaling.

- **Reinforcement Learning (RLVR)** — [kernelbench-tinker](https://github.com/ScalingIntelligence/kernelbench-tinker) is an end-to-end integration with Thinking Machines Lab's [Tinker RL library](https://github.com/thinking-machines-lab/tinker). The pipeline has the policy model generate kernels, evaluates them on cloud GPUs via Modal, and converts results into RL rewards — a minimal playground for experimenting with RLVR on GPU kernel optimization.

- **Evolutionary Search** — Evolutionary search like AlphaEvolve has shown promise for discovering innovative solutions for optimization problems. We are working on an integration for [OpenEvolve](https://github.com/algorithmicsuperintelligence/openevolve). Releasing soon.

- **Roofline / Max Speedup Analysis** —   *(Experimental, WIP)* [simple-torchroofline](https://github.com/simonguozirui/simple-torchroofline) provides analytical roofline analysis for PyTorch programs, estimating the speed-of-light (SoL) compute and memory bounds for a target GPU — no hardware required. Combined with hardware counter-based profiling for empirical roofline analysis, this helps sanity-check whether a reported speedup is physically realistic. 


## 🔍 Known Usage
Since release, we have gotten a lot of interest from researchers, research labs, and companies that use KernelBench to explore this direction. We have documented [known usage](https://docs.google.com/document/d/e/2PACX-1vTjS-UMH1HB5n_PENq2k-3YRfXIXkqKIKeNC2zcWMyLPdl4Jrwvdk4dNDVSsM8ybKrCxZB7GJq1slZF/pub) of KernelBench and related efforts towards automated kernel generations. If you are using KernelBench, we love to hear more about it!

Disclaimer: KernelBench is designed as an **open-source** evaluation framework and toolkit. The KernelBench team does not review, validate, or endorse individual kernels or reported results. Users are responsible for independently verifying any results obtained using the framework. Please check out `EVAL.md` for more guidance on benchmarking and evaluating kernels.


## 🪪 License
MIT. Check `LICENSE.md` for more details.


## Citation
```bibtex
@misc{ouyang2025kernelbenchllmswriteefficient,
      title={KernelBench: Can LLMs Write Efficient GPU Kernels?}, 
      author={Anne Ouyang and Simon Guo and Simran Arora and Alex L. Zhang and William Hu and Christopher Ré and Azalia Mirhoseini},
      year={2025},
      eprint={2502.10517},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.10517}, 
}
```

We are grateful for the support from [GPU Mode](https://gpu-mode.github.io/popcorn/), [PyTorch](https://pytorch.org/), [Modal Labs](https://modal.com/blog/accelerating-ai-research-case-study) and the broader open-source community that made this project possible.
