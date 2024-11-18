# KernelBench

A benchmark for evaluating LLMs' ability to generate GPU kernels

![KernelBenchMascot](./assets/figures/KernelBenchMascot.png)



## Task Description

We construct Kernel Bench to have 3 Levels of difficulty:
- Level 1: Single-kernel operators (100 Problems)
    The foundational building blocks of neural nets (Convolutions, Matrix multiplies, Layer normalization)
- Level 2: Simple fusion patterns (100 Problems)
    A fused kernel would be faster than separated kernels (Conv + Bias + ReLU, Matmul + Scale + Sigmoid)
- Level 3: Full model architectures (50 Problems)
    Optimize entire model architectures end-to-end (MobileNet, VGG, MiniGPT, Mamba)

So far we have curated 250 problems across all levels..

## Directory Structure
TODO: Update this
```
KernelBenchInternal/
├── assets/
├── KernelBench/
├── src/
│   ├── analysis.py
│   ├── eval.py
│   └── utils/
├── 
```

## Set up
```
conda create --name kernel-bench python=3.10
conda activate kernel-bench
pip install -r requirements.txt
# set up the repo
pip install -e . 
```

Set your `{INFERENCE_SERVER_PROVIDER}_API_KEY` API key.

Running and profiling kernels require a GPU. If you don't have GPU available locally, you can set up [Modal](https://modal.com/). Set up your modal token.

## How to use
Run Eval
You can set up a database to go with this or you can write to a local JSON file.