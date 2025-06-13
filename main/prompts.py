"""
Prompt Templates for test-time scaling methods
"""
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_iterative_refinement

from configs import TestTimeScalingConfig
from utils import WorkArgs, fetch_kernel_from_disk, fetch_eval_result_from_disk


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
    pass


def generate_prompt_cognition(work: WorkArgs, config: TestTimeScalingConfig, ref_arch_src: str, run_dir: str) -> str:
    pass


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
        case "Cognition":
            return generate_prompt_cognition(work, config, ref_arch_src, run_dir)
        case "Stanford":
            return generate_prompt_stanford(work, config, ref_arch_src, run_dir)
        case _:
            raise ValueError(f"Invalid method: {config.method}")
