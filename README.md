# KernelBench
[blog post](https://scalingintelligence.stanford.edu/) | [dataset](https://huggingface.co/datasets/kernelbench)

A benchmark for evaluating LLMs' ability to generate GPU kernels

![KernelBenchMascot](./assets/figures/KernelBenchMascot.png)

<!-- TODO: Add blog post -->

See [blog post](https://scalingintelligence.stanford.edu/) for more details.

## 👋 Task Description
*NEED ANNE FEEDBACK*
We structure the problem for LLM to transpile operators described in PyTorch to CUDA kernels, at whatever level of granularity.
<!-- ADD A DIAGRAM -->

We construct Kernel Bench to have 4 Levels of categories:
- Level 1: Single-kernel operators (100 Problems)
    The foundational building blocks of neural nets (Convolutions, Matrix multiplies, Layer normalization)
- Level 2: Simple fusion patterns (100 Problems)
    A fused kernel would be faster than separated kernels (Conv + Bias + ReLU, Matmul + Scale + Sigmoid)
- Level 3: Full model architectures (50 Problems)
    Optimize entire model architectures end-to-end (MobileNet, VGG, MiniGPT, Mamba)
- Level 4: Level Hugging Face
    Optimize whole model architectures from HuggngFace

*TODO: JUSTIFY CRITIERA*
We care whether if a solution 
- compiles: (WDYM BY COMPILE)
- is correct: explain correctness criteria (how do we check)
- is fast: we compare against baseline (how many times)

## 🔍 Directory Structure
We organize the repo into the following structure:
```
KernelBenchInternal/
├── assets/
├── KernelBench/
├── src/
│   ├── tests/  
│   ├── prompts/
│   ├── other files
├── scripts/
├── results/
```

## 🔧 Set up
```
conda create --name kernel-bench python=3.10
conda activate kernel-bench
pip install -r requirements.txt
pip install -e . 
```

To call LLM API providers, set your `{INFERENCE_SERVER_PROVIDER}_API_KEY` API key.

Running and profiling kernels require a GPU. If you don't have GPU available locally, you can set up [Modal](https://modal.com/). Set up your modal token.

## 🚀 Usage
### Run on a single problem 
It is easier to debug with a single problem.
```
python3 scripts/generate_and_eval_single_sample.py level=1 problem_id=0
```

### Run on all problems 

```
# 1. Generate responses and store them locally
python3 scripts/generate_samples.py level=1

# 2. Evaluate on all problems
python3 scripts/eval_from_generation.py level=1
```

We provide some reference baseline times on NVIDIA L40S in `results/timing` (soon also on H100).

## 🛣️ Upcoming Roadmap
- [ ] More reference baseline times on various GPU platforms
- [ ] Easy-to-useCloud GPU Integration (Modal)
- [ ] Integrate with more frameworks, such as [ThunderKittens](https://github.com/HazyResearch/ThunderKittens)
- [ ] Add backward pass
- [ ] Integrate with toolchains such as NCU


<!-- Add Citation -->

<!-- Add License -->
## 🪪 License
MIT. Check `LICENSE.md` for more details.