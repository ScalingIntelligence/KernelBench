# import time
# from tqdm import tqdm
# from src import eval, utils
# import torch
# import os
# import multiprocessing as mp

import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_file_one_example
from src.utils import extract_first_code, query_server, set_gpu_arch
from src.run import run_llm

"""
Make this an all-in-one script for now
Eval Script using Hugging Face Datasets

Heavily inspired by 
https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/harness/run_evaluation.py


This is a simple version, I will ship the more compelx version when I get the others one fixed up
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class EvalConfig(Config):
    def __init__(self):
        # name of dataset name on Hugging Face
        self.dataset_name = REQUIRED

        self.dataset_name = "anneouyang/kbtest"

        # TODO to decide
        # self.split = "test"
        # others: cache, build

        self.log = True
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")

        # where we are runing eval
        # local (requires a GPU), modal?
        self.eval_mode = "local"

        self.level = 1

        # let's just eval 1 problem right now!
        # NOTE: this is the logical index (problem id the problem_name)
        self.problem_id = 1

        # Inference
        self.max_tokens = 4096
        self.server_type = "deepseek"

        # enforce for now
        self.num_workers = 1

        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]
        
        self.temperature = 0.7


    def greedy(self):
        self.temperature = 0.0

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    For this design let's run all the steps together!
    We will think abotu the inference + eval separation later (maybe like swe bench or cuda_monkey style)

    """
    print(f"Starting Eval with config: {config}")

    dataset = load_dataset(config.dataset_name)

    if config.gpu_arch:
        set_gpu_arch(config.gpu_arch)  # otherwise build for all architectures

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # just make it simple for now
    curr_level_dataset = dataset[f"level_{config.level}"]

    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"

    problem_idx_in_dataset = config.problem_id - 1 # due to dataset being 0-indexed

    # 1. fetch reference architecture from problem directory
    ref_arch_src = curr_level_dataset[problem_idx_in_dataset]["code"]
    problem_name = curr_level_dataset[problem_idx_in_dataset]["name"]

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number <= config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    # 2. generate samples

    # Create inference function with config parameters
    inference_fn = lambda prompt: run_llm(
        prompt,
        server_type=config.server_type,
        temperature=config.temperature, 
    )

    custom_cuda_prompt = prompt_generate_custom_cuda_from_file_one_example(ref_arch_src, 1)
    # if save_prompt:
    #     with open(os.path.join(REPO_TOP_PATH, "src/scratch/prompt.txt"), "w") as f:
    #         f.write(custom_cuda_prompt)

    custom_cuda = inference_fn(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, "python")
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    
    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"model_new_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)

    # 3. evaluate kernel against reference architecture
    # need to wrap around process, see test_eval.py
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src, custom_cuda, verbose=False, measure_performance=True
    )

    # NOTE: should I replace this with a json file? rather than just a text log?
    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))

            

if __name__ == "__main__":
    main()

