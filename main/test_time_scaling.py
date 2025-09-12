import yaml
import os
import torch
import multiprocessing as mp
from datasets import load_dataset
import wandb
from llm_utils import create_llm_client

from dotenv import load_dotenv
load_dotenv()
import sys
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_TOP_DIR)

from src.utils import set_gpu_arch, WorkArgs
from src.dataset import construct_kernelbench_dataset, fetch_ref_arch_from_level_problem_id

from main.configs import parse_test_time_scaling_args, RUNS_DIR
from main.generation_utils import batch_generate
from main.evaluation_utils import batch_eval

"""
Implements basic test-time scaling approaches
1. best-of-N
2. iterative refinement
3. METR evolutionary approach
4. Stanford: NL idea gen + branching (TODO)
"""

def base(config, level, problem_id_range: range, llm_client: callable, run_dir: str):
    """
    Base approach
    """
    workload = []

    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        workload.append(
            WorkArgs(
                level=level,
                problem_id=int(problem_id),
                sample_id=0
            )
        )
    
    batch_generate(workload, config, llm_client, run_dir)    
    
    eval_file_path = os.path.join(run_dir, f"eval_results.json")
    batch_eval(workload, config, run_dir, eval_file_path)


def best_of_n(config, level, problem_id_range: range, llm_client: callable, run_dir: str):
    """
    Best-of-N approach
    Generate num_samples for each problem independently
    """
    # Define workloads
    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    for sample_id in range(config.num_parallel):
        workload = []
        for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
            workload.append(
                WorkArgs(
                    level=level,
                    problem_id=int(problem_id),
                    sample_id=sample_id
                )
            )
        
        batch_generate(workload, config, llm_client, run_dir)     
        batch_eval(workload, config, run_dir, eval_file_path) 



def iterative_refinement(config, level, problem_id_range: range, llm_client: callable, run_dir: str, rule_path=None):
    """
    Iterative refinement approach
    """
    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    num_iterations = config.num_iterations
    for iteration in range(num_iterations):
        print(f"[Iterative Refinement] Iteration {iteration + 1} of {num_iterations}")
        
        # Generate samples
        workload = []
        for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
            for sample_id in range(config.num_parallel):
                workload.append(
                    WorkArgs(
                        level=level,
                        problem_id=int(problem_id),
                        sample_id=sample_id + iteration * config.num_parallel
                    )
                )

        batch_generate(workload, config, llm_client, run_dir, rule_path)
        batch_eval(workload, config, run_dir, eval_file_path)


def metr(config, level, problem_id_range: range, llm_client: callable, run_dir: str):
    """
    METR approach
    1. Generate 8 samples in parallel
    2. When a thread is done, sample from currently evaluated kernels based on efficiency
    3. Generate new attempt based on the sample 
    4. Repeat until num_samples are generated
    """
    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    # 0. Add the reference architecture as the first sample
    print(f"[METR] Adding reference architecture as the first sample")
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem_id, config.dataset_src)
        kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{0}_kernel.py")
        with open(kernel_path, "w") as f:
            f.write(ref_arch_src)

    workload = []
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        workload.append(
            WorkArgs(
                level=level,
                problem_id=int(problem_id),
                sample_id=0
            )
        )
    
    batch_eval(workload, config, run_dir, eval_file_path) 

    # 1. Generate 8 samples in parallel
    print(f"[METR] Generating {config.num_parallel} samples in parallel")
    workload = []
    for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
        for sample_id in range(1, config.num_parallel + 1):
            workload.append(
                WorkArgs(
                    level=level,
                    problem_id=int(problem_id),
                    sample_id=sample_id
                )
            )
    
    batch_generate(workload, config, llm_client, run_dir)
    batch_eval(workload, config, run_dir, eval_file_path)

    # 2. Continue generating samples until we reach num_samples
    for sample_id in range(config.num_parallel + 1, config.num_samples + 1):
        print(f"[METR] Generating sample {sample_id} of {config.num_samples}")
        workload = []
        for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
            workload.append(
                WorkArgs(
                    level=level,
                    problem_id=int(problem_id),
                    sample_id=sample_id
                )
            )
            
        batch_generate(workload, config, llm_client, run_dir)
        batch_eval(workload, config, run_dir, eval_file_path)


def stanford(config, level, problem_id_range: range, llm_client: callable, run_dir: str):
    """
    Stanford approach: Beam Search variant
    """
    eval_file_path = os.path.join(run_dir, f"eval_results.json")

    for iteration in range(config.num_iterations):
        print(f"[Stanford] Iteration {iteration + 1} of {config.num_iterations}")
        
        workload = []
        for problem_id in range(problem_id_range.start, problem_id_range.stop + 1): # end index is inclusive
            for sample_id in range(config.num_parallel):
                workload.append(
                    WorkArgs(
                        level=level,
                        problem_id=int(problem_id),
                        sample_id=sample_id + iteration * config.num_parallel
                    )
                )
        
        batch_generate(workload, config, llm_client, run_dir)
        batch_eval(workload, config, run_dir, eval_file_path)


def main(config):
    """
    Test-Time Scaling for Particular Level
    """
    tags = ["test-time-scaling"] + config._tags.split(",")
    tags.extend([config.run_name, config.method, config.prompt, str(config.level)])
    wandb.init(
        project="KernelBench",
        entity="j1mk1m",
        tags=tags
    )
    wandb.log({"run_name": config.run_name, "method": config.method, "prompt": config.prompt, "level": config.level, "model_name": config.model_name})
    print(f"Starting Test-Time Scaling with config: {config}")

    # Check if CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available. Evaluation requires GPU.")

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")

    # 1. Set up
    # Set up dataset
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Level {config.level} problems: {problem_id_range}")

    # set up run directory
    run_dir = os.path.join(RUNS_DIR, config.run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(config), f)
 
    assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

    # set GPU arch to configure what target to build for
    set_gpu_arch(config.gpu_arch)
    assert config.num_eval_devices <= torch.cuda.device_count(), f"Number of GPUs requested ({config.num_eval_devices}) is greater than the number of available GPUs ({torch.cuda.device_count()})"

    # Create inference function with config parameters
    default_base_api = f"http://{config.vllm_host}:{config.vllm_port}/v1" if config.server_type == "vllm" else None
    llm_client = create_llm_client(os.path.join(run_dir, "llm_usage.json"),
                                   default_model_name=config.model_name,
                                   default_base_api=default_base_api,
                                   default_temperature=config.temperature,
                                   default_max_tokens=config.max_tokens)
    

    # Run the test-time scaling approach
    match config.method:
        case "base":
            base(config, config.level, problem_id_range, llm_client, run_dir)
        case "best-of-N":
            best_of_n(config, config.level, problem_id_range, llm_client, run_dir)
        case "iterative refinement":
            iterative_refinement(config, config.level, problem_id_range, llm_client, run_dir)
        case "METR":
            metr(config, config.level, problem_id_range, llm_client, run_dir)
        case "Stanford":
            stanford(config, config.level, problem_id_range, llm_client, run_dir)
        case _:
            raise ValueError(f"Invalid method: {config.method}")
 

if __name__ == "__main__": 
    args = parse_test_time_scaling_args()
    main(args)

