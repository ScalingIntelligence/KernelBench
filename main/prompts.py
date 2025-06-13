"""
Prompt Templates for test-time scaling methods
"""
import os
import random

from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_iterative_refinement
from src.utils import read_file

from configs import TestTimeScalingConfig
from utils import WorkArgs, fetch_kernel_from_disk, fetch_eval_result_from_disk, get_evaluation_results_for_problem


def generate_prompt_best_of_n(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, run_dir: str) -> str:
    # Default prompt
    return prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)


def generate_prompt_iterative_refinement(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, run_dir: str) -> str:
    if work.sample_id < config.num_samples:
        return prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    
    # Fetch last generated kernel and execution results
    last_kernel_src = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, work.sample_id - config.num_samples)
    last_exec_result = fetch_eval_result_from_disk(run_dir, config.level, work.problem_id, work.sample_id - config.num_samples)
    
    # Construct prompt
    prompt = prompt_iterative_refinement(ref_arch_src, last_kernel_src, last_exec_result)
    
    return prompt


def generate_prompt_metr(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, run_dir: str) -> str:
    if work.sample_id <= config.num_parallel:
        return prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    
    # Sample from previously generated kernels based on efficiency
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    eval_results = get_evaluation_results_for_problem(work.problem_id, eval_file_path)

    ref_kernel_result = eval_results["0"]
    assert ref_kernel_result["correctness"], "Reference kernel is not correct"

    correct_kernels = [eval_result for eval_result in eval_results.values() if eval_result["correctness"]]
    
    # Sample from the correct kernels based on efficiency
    speedups = [ref_kernel_result["runtime"] / eval_result["runtime"] for eval_result in correct_kernels]
    sampled_kernel_eval_result = random.choices(correct_kernels, weights=speedups)[0]
    sampled_kernel_id = int(sampled_kernel_eval_result["sample_id"])
    if config.verbose:
        print(f"[METR] Sampled kernel {sampled_kernel_id} with speedup {ref_kernel_result['runtime'] / sampled_kernel_eval_result['runtime']}")

    sampled_kernel_src = fetch_kernel_from_disk(run_dir, config.level, work.problem_id, sampled_kernel_id)

    return prompt_iterative_refinement(ref_arch_src, sampled_kernel_src, sampled_kernel_eval_result)


def generate_prompt_stanford(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, run_dir: str) -> str:
    pass


def generate_prompt(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, run_dir: str) -> str:
    match config.method:
        case "best-of-N":
            return generate_prompt_best_of_n(work, config, ref_arch_src, run_dir)
        case "iterative refinement":
            return generate_prompt_iterative_refinement(work, config, ref_arch_src, run_dir)
        case "METR":
            return generate_prompt_metr(work, config, ref_arch_src, run_dir)
        case "Stanford":
            return generate_prompt_stanford(work, config, ref_arch_src, run_dir)
        case _:
            raise ValueError(f"Invalid method: {config.method}")
