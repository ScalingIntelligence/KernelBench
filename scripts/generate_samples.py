import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
from dataclasses import dataclass


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

class GenerationConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED
        
        # subset of problems to generate on all problems in the level
        self.subset = (None, None) # (problem_id, problem_name), these are the logical index

        self.run_name = REQUIRED # name of the run

        # num of thread pool to call inference server in parallel
        self.num_workers = 1
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0
        
        # Logging
        # Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
    
        self.verbose = False

        self.store_type = "local" # TODO: add Database Integration

        # Future support
        # Migrate Monkeys code base to KernelBench
        # self.num_samples = 0 # for sampling multiple samples per problem

        # logging verbosity level#  # TODO
        # self.log = False
        # self.log_prompt = False
        # self.log_generated_kernel = False

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"
    

@dataclass
class WorkArgs:
    problem_id: int # logically indexed
    sample_id: int

def generate_sample(work, config, dataset, inference_server):
    print(f"Generating sample {work.problem_id} {work.sample_id}")
    pass    

def generate_sample_launcher(work, config, dataset, inference_server):
    try:
        return generate_sample(work, config, dataset, inference_server)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    print(f"Starting Batch Generation with config: {config}")

    # Dataset Configurations
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)


    num_problems_in_level = len(curr_level_dataset)

    if config.subset == (None, None):
        problem_id_range = range(1, num_problems_in_level)
    else:
        assert config.subset[1] <= num_problems_in_level, f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(f"Generating on 1 sample each for level {config.level} problems {problem_id_range}")

    # set up run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)


    assert config.store_type == "local", "supporting local file-system based storage for now" # database integreation coming soon, need to migrate from CUDA Monkeys code

    problems_to_run = []
    for problem_id in problem_id_range:
        
        # TODO: check if already exist 
        problems_to_run.append(
            WorkArgs(
                problem_id=int(problem_id),
                sample_id=0 # fix to 0 for now
            )
        )



    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose)
    

    maybe_multithread(generate_sample_launcher, problems_to_run, config.num_workers, time_interval=config.api_query_interval, config=config, dataset=curr_level_dataset, inference_server=inference_server)

    return

        


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



    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)
    
    


if __name__ == "__main__":
    main()

