# KernelBench - Can LLMs Write GPU Kernels?
[blog post](https://scalingintelligence.stanford.edu/blogs/kernelbench/) | [dataset](https://huggingface.co/datasets/ScalingIntelligence/KernelBench)

A benchmark for evaluating LLMs' ability to generate GPU kernels

![KernelBenchMascot](./assets/figures/KernelBenchMascot.png)

<!-- TODO: Add blog post -->

See [blog post](https://scalingintelligence.stanford.edu/blogs/kernelbench/) for more details.

## 👋 Task Description
We structure the problem for LLM to transpile operators described in PyTorch to CUDA kernels, at whatever level of granularity it desires to.
![KernelBenchMascot](./assets/figures/KernelBenchWorkFlow.png)

We construct Kernel Bench to have 4 Levels of categories:
- Level 1: Single-kernel operators (100 Problems)
    The foundational building blocks of neural nets (Convolutions, Matrix multiplies, Layer normalization)
- Level 2: Simple fusion patterns (100 Problems)
    A fused kernel would be faster than separated kernels (Conv + Bias + ReLU, Matmul + Scale + Sigmoid)
- Level 3: Full model architectures (50 Problems)
    Optimize entire model architectures end-to-end (MobileNet, VGG, MiniGPT, Mamba)
- Level 4: Level Hugging Face
    Optimize whole model architectures from HuggngFace

For this benchmark, we care whether if a solution 
- **compiles**: generated torch code was able to load the inline embedded CUDA Kernel and build the kernel
- **is correct**: check against reference torch operators n_correctness times on randomized inputs
- **is fast**: compare against reference torch operators n_trial times for both eager mode and torch.compile execution

## 🔍 Directory Structure
We organize the repo into the following structure:
```
KernelBench/
├── assets/
├── KernelBench/ # Benchmark dataset files
├── src/ # KernelBench logic code
│   ├── unit_tests/  
│   ├── prompts/
│   ├── ....
├── scripts/ # helpful scripts to run the benchmark
├── results/ # baseline times across hardware 
├── runs/ # where your runs will be stored
```

## 🔧 Set up
```
conda create --name kernel-bench python=3.10
conda activate kernel-bench
pip install -r requirements.txt
pip install -e . 
```

To call LLM API providers, set your `{INFERENCE_SERVER_PROVIDER}_API_KEY` API key.

Running and profiling kernels require a GPU. 
If you don't have GPU available locally, you can set up [Modal](https://modal.com/). Set up your modal token after creating an account by running `modal token new`. Then, use the `generate_and_eval_single_sample_modal.py` script.

## 🚀 Usage
### Run on a single problem 
It is easier to get started with a single problem. This will fetch the problem, generate a sample, and evaluate the sample.

```
# for example, run level 2 problem 40 from huggingface

python3 scripts/generate_and_eval_single_sample.py dataset_src="huggingface" level=2 problem_id=40

# dataset_src could be "local" or "huggingface"
# add .verbose_logging for more visbility
```

### Run on all problems 

```
# 1. Generate responses and store kernels locally to runs/{run_name} directory
python3 scripts/generate_samples.py run_name="test_hf_level_1" dataset_src="huggingface" level="1" num_workers=50 server_type="deepseek" model_name="deepseek-coder" temperature=0

# 2. Evaluate on all generated kernels in runs/{run_name} directory
python3 scripts/eval_from_generations.py level=1 run_name="test_hf_level_1" dataset_src="local" level="1" num_gpu_devices=8 timeout=300

```

You can check out `scripts/greedy_analysis.py` to analyze the eval results.
We provide some reference baseline times a variety of NVIDIA GPUs across generations in `results/timing`.

## 🛣️ Upcoming Roadmap
- [ ] Integrate with more frameworks, such as [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- [ ] Add backward pass
- [ ] Integrate with toolchains such as NCU


<!-- Add Citation -->

## 🪪 License
MIT. Check `LICENSE.md` for more details.

## Citing
```bibtex
@misc{ouyang2024kernelbench,
      title={KernelBench: Can LLMs Write GPU Kernels?}, 
      author={Anne Ouyang and Simon Guo and Azalia Mirhoseini},
      year={2024},
      url={https://scalingintelligence.stanford.edu/blogs/kernelbench/}, 
}
```
