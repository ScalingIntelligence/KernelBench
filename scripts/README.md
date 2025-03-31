# KernelBench Scripts Guide

This directory contains scripts for generating GPU kernels using LLMs and evaluating their performance compared to PyTorch baselines.

## Script Categories

The scripts can be organized into several categories:

1. **Generation scripts** - Generate GPU kernels using LLMs
2. **Evaluation scripts** - Evaluate kernel correctness and performance
3. **Analysis scripts** - Analyze evaluation results
4. **Inspection/Debugging scripts** - Tools for debugging and inspecting kernels
5. **Modal variants** - Cloud-based versions of scripts using Modal

## Core Scripts

### Generation Scripts

- **`generate_samples.py`** - Generate kernels for multiple problems
  ```bash
  # Example: Generate kernels for all level 1 problems using the DeepSeek model
  python generate_samples.py run_name=test_hf_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0
  ```

- **`generate_and_eval_single_sample.py`** - Generate and evaluate a kernel for a single problem
  ```bash
  # Example: Generate and evaluate a kernel for level 2, problem 40
  python generate_and_eval_single_sample.py dataset_src="huggingface" level=2 problem_id=40
  ```

### Evaluation Scripts

- **`run_and_check.py`** - Evaluate a generated kernel against a reference implementation
  ```bash
  # Example: Evaluate a generated kernel
  python run_and_check.py --kernel_path=path/to/kernel.py --reference_path=path/to/reference.py
  ```

- **`eval_from_generations.py`** - Evaluate all generated kernels in a run directory
  ```bash
  # Example: Evaluate all kernels from a previous generation run
  python eval_from_generations.py run_name=test_hf_level_1 dataset_src=local level=1 num_gpu_devices=8 timeout=300
  ```

- **`verify_generation.py`** - Verify if a generated kernel is correct
  ```bash
  # Example: Verify a kernel's correctness
  python verify_generation.py --kernel_path=path/to/kernel.py --reference_path=path/to/reference.py
  ```

### Analysis Scripts

- **`benchmark_eval_analysis.py`** - Analyze evaluation results to compute benchmark metrics
  ```bash
  # Example: Analyze results from evaluation
  python benchmark_eval_analysis.py run_name=test_hf_level_1 level=1 hardware=L40S_matx3 baseline=baseline_time_torch
  ```

- **`generate_baseline_time.py`** - Generate baseline timing results for PyTorch implementations
  ```bash
  # Example: Generate baseline timing for level 1 problems
  python generate_baseline_time.py level=1 run_name=baseline_torch_l1 n_trials=100
  ```

### Inspection Scripts

- **`inspect_baseline.py`** - Inspect baseline PyTorch implementation details
  ```bash
  # Example: Inspect baseline for a specific problem
  python inspect_baseline.py level=1 problem_id=10
  ```

- **`inspect_triton.py`** - Inspect Triton kernel implementation details
  ```bash
  # Example: Inspect Triton kernel for a specific problem
  python inspect_triton.py level=1 problem_id=10
  ```

- **`inspect_kernel_pytorch_profiler.py`** - Profile kernels with PyTorch profiler
  ```bash
  # Example: Profile a kernel with PyTorch profiler
  python inspect_kernel_pytorch_profiler.py --kernel_path=path/to/kernel.py
  ```

## Modal Variants

These scripts use [Modal](https://modal.com/) for cloud-based execution:

- **`generate_and_eval_single_sample_modal.py`** - Cloud version of single sample generation/evaluation
  ```bash
  # Example: Generate and evaluate a kernel on Modal
  python generate_and_eval_single_sample_modal.py dataset_src="huggingface" level=2 problem_id=40
  ```

- **`generate_baseline_time_modal.py`** - Cloud version of baseline timing generation
  ```bash
  # Example: Generate baseline timing on Modal
  python generate_baseline_time_modal.py level=1 run_name=baseline_torch_l1_modal n_trials=100
  ```

- **`run_and_check_modal.py`** - Cloud version of kernel evaluation
  ```bash
  # Example: Evaluate a kernel on Modal
  python run_and_check_modal.py --kernel_path=path/to/kernel.py --reference_path=path/to/reference.py
  ```

- **`server_run_and_check.py`** and **`server_run_and_check_modal.py`** - Server variants for continuous evaluation

## Workflow Examples

### Complete Local Workflow

1. Generate kernels for all level 1 problems:
   ```bash
   python generate_samples.py run_name=test_level_1 dataset_src=huggingface level=1 num_workers=50 server_type=deepseek model_name=deepseek-chat temperature=0
   ```

2. Evaluate the generated kernels:
   ```bash
   python eval_from_generations.py run_name=test_level_1 dataset_src=local level=1 num_gpu_devices=8 timeout=300 build_cache=True num_cpu_workers=16
   ```

3. Analyze the results:
   ```bash
   python benchmark_eval_analysis.py run_name=test_level_1 level=1 hardware=L40S_matx3 baseline=baseline_time_torch
   ```

### Cloud-based Single Problem Workflow

1. Set up Modal:
   ```bash
   modal token new
   ```

2. Generate and evaluate a kernel on Modal:
   ```bash
   python generate_and_eval_single_sample_modal.py dataset_src="huggingface" level=2 problem_id=40
   ```

## Note on Code Reuse

There is significant opportunity for code reuse and consolidation between the standard and Modal versions of scripts. Consider refactoring to create a common core library that both local and cloud variants can leverage.
