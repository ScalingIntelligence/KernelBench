import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, set_gpu_arch, read_file, create_inference_server_from_presets, maybe_multithread

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        # self.problem_id
        
        self.run_name = REQUIRED

        self.num_workers = 1
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_generated_kernel = False
        self.log_eval_result = False

        # Store 
        # either locally or on disk
        self.store_type = "local"

        # Future support
        # Migrate Monkeys code base to KernelBench
        # self.num_samples = 0

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True
        
    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Keep it simple: Generate and evaluate a single sample
    """
    print(f"Starting Eval with config: {config}")

    # Configurations

    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems_in_dataset = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems_in_dataset}")


    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    # assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # 1. Fetch Problem
    if config.dataset_src == "huggingface":

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    # import pdb; pdb.set_trace()

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    
    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose, 
                                                        time_generation=True)
    


    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"


    # psuedo code
    # # need to crate to_run lists
    # make run query and record result!
    # need to figure out how to store files locally 
    # ideally similar to monkeys code base

    maybe_multithread(run, to_run, config.num_workers, time_interval=config.api_query_interval, config=config)


    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)
    
    


if __name__ == "__main__":
    main()

