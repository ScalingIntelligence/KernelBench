# from dataclasses import dataclass
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

"""
Make this an all-in-one script for now
Eval Script using Hugging Face Datasets

Heavily inspired by 
https://github.com/princeton-nlp/SWE-bench/blob/main/swebench/harness/run_evaluation.py
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

# dataset = utils.construct_problem_dataset_from_problem_dir(PROBLEM_DIR)

# # NOTE: If you run this this will have cascading errors from previous iterations
# # # evaluate kernel (for some samples)
# # for sample_id in range(1,5):
# #     print(f"Evaluating for sample {sample_id}")
# #     print(f"[Curr Eval] Evaluating Kernel for Run {RUN_NAME} on Problem {problem_id} with Sample {sample_id} on CUDA device {device}: {torch.cuda.get_device_name(device)}")

# #     # fetch reference architecture from problem directory
# #     ref_arch_src = eval.fetch_ref_arch_from_problem_id(problem_id, dataset)

# #     # fetch kernel code from database
# #     kernel_src = eval.fetch_kernel_from_database(RUN_NAME, problem_id, sample_id, SERVER_URL)
# #     assert kernel_src is not None, f"Kernel not found for sample {sample_id}"

# #     try:
# #         eval_result = eval.eval_kernel_against_ref(original_model_src=ref_arch_src,
# #                                                custom_model_src=kernel_src,
# #                                                measure_performance=MEASURE_PERFORMANCE,
# #                                                verbose=True,
# #                                                device=device)
# #         print("-" * 32)
# #         print(f"Eval result for sample {sample_id}: {eval_result}")
# #         print("-" * 32)
# #     except Exception as e:
# #         print(f"THIS SHOULD NOT PRINT for sample {sample_id}: Some issue evaluating for kernel: {e} ")
# #     finally:
# #         torch.cuda.empty_cache()


# @dataclass
# class WorkArgs:
#     problem_id: str
#     sample_idx: int
#     run_name: str
#     dataset: list[str]
#     device: torch.device


# def run(work, config=None, coordinator=None):
#     """
#     Matching Monkey API, took out some args for config and coordinator
#     """
#     run_inner = evaluate_single_sample
#     # if config.testing: return run_inner(work)
#     try:
#         eval_result = run_inner(work)
#         with open(f"results/eval_result_problem_{work.problem_id}.txt", "a") as f:
#             f.write("-" * 128 + "\n")
#             f.write(f"Eval result for sample {work.sample_idx}: {eval_result}\n")

#         print("-" * 32)
#         print(
#             f"Eval result for problem {work.problem_id} sample {work.sample_idx}: {eval_result}"
#         )
#         print("-" * 32)
#     except Exception as e:
#         print("Error", e, work.problem, work.problem_id, work.sample_idx)
#         return None


# def evaluate_single_sample(work_args: WorkArgs, configs: dict):
#     # problem_id, sample_id, run_name, dataset, device
#     problem_id, sample_id, run_name, dataset, device = (
#         work_args.problem_id,
#         work_args.sample_idx,
#         work_args.run_name,
#         work_args.dataset,
#         work_args.device,
#     )
#     num_correct_trials = configs["num_correct_trials"]
#     num_perf_trials = configs["num_perf_trials"]
#     verbose = configs["verbose"]
#     measure_performance = configs["measure_performance"]
#     # fetch reference architecture from problem directory
#     ref_arch_src = eval.fetch_ref_arch_from_problem_id(problem_id, dataset)

#     # fetch kernel code from database
#     kernel = eval.fetch_kernel_from_database(
#         run_name, problem_id, sample_id, SERVER_URL
#     )
#     kernel_src = kernel["kernel"]
#     kernel_hash = kernel["kernel_hash"]
#     assert kernel_src is not None, f"Kernel not found for sample {sample_id}"
#     try:
#         eval_result = eval.eval_kernel_against_ref(
#             original_model_src=ref_arch_src,
#             custom_model_src=kernel_src,
#             custom_model_hash=kernel_hash,
#             measure_performance=measure_performance,
#             verbose=verbose,
#             num_correct_trials=num_correct_trials,
#             num_perf_trials=num_perf_trials,
#             # move this to config in monkeys
#             build_dir=f"/matx/u/simonguo/kernel_eval_build/{run_name}/{problem_id}/{sample_id}",
#             device=device,
#         )
#         return eval_result
#     except Exception as e:
#         print(
#             f"[WARNING] Last level catch on {sample_id}: Some issue evaluating for kernel: {e} "
#         )
#         if "CUDA error" in str(e):
#             # NOTE: count this as compilation failure as it is not runnable code
#             metadata = {
#                 "cuda_error": f"CUDA Error: {str(e)}",
#                 "hardware": torch.cuda.get_device_name(device=device),
#                 "device": device,
#             }  # for debugging
#             eval_result = eval.KernelExecResult(
#                 compiled=False, correctness=False, metadata=metadata
#             )
#             return eval_result
#         return None


class EvalConfig(Config):
    def __init__(self):
        # name of dataset name on Hugging Face
        self.dataset_name = REQUIRED

        self.dataset_name = "anneouyang/kbtest"

        # TODO to decide
        # self.split = "test"
        # others: cache, build

        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")

        # where we are runing eval
        # local (requires a GPU), modal?
        self.eval_mode = "local"

        self.level = "level_1"

        # let's just eval 1 problem right now!
        self.problem_id = 0

        # enforce for now
        self.num_workers = 1

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    For this design let's run all the steps together!
    let's just do one right now!
    """
    print(f"Starting Eval with config: {config}")

    dataset = load_dataset(config.dataset_name)

    # just make it simple for now
    curr_level_dataset = dataset[config.level]

    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")

    assert config.problem_id < len(
        curr_level_dataset
    ), f"Problem ID {config.problem_id} out of range for Level {config.level}"

    # 1. fetch reference architecture from problem directory
    ref_arch_src = curr_level_dataset[config.problem_id]["code"]
    problem_name = curr_level_dataset[config.problem_id]["name"]
    # 2. generate samples
    import pdb

    pdb.set_trace()

    # 3. fetch kernel code from database

    # 3. evaluate kernel against reference architecture


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     # problem_id = 7
#     # samples_range = (4, 5)
#     # problem_range = (15, 54)
#     # samples_range = (2, 10) # 30 samples

#     # problem_range = (15, 16)
#     # samples_range = (0, 1)
#     # Check if CUDA is available
#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA device not available. This test requires a GPU.")

#     # these will go into pydra in the future
#     configs = {
#         "num_correct_trials": 5,
#         "num_perf_trials": 100,
#         "timeout": 20,
#         "verbose": False,
#         "num_gpu_devices": NUM_GPU_DEVICES,
#         "measure_performance": True,
#     }

#     problem_range = (3, 4)
#     samples_range = (0,)

#     # this works great, launch process one at a time
#     # cuda_eval_process(problem_range, samples_range, configs)

#     # batch eval, in our experiment server it will be replaced by fetching from database
#     total_work = []  # a list of (problem_id, sample_id)
#     for problem_id in range(*problem_range):
#         for sample_id in range(*samples_range):
#             kernel_id = -1  # fake example
#             total_work.append((problem_id, sample_id, kernel_id))

#     # # this does it in a batch manner
#     batch_eval(total_work, RUN_NAME, dataset, configs)

#     # use this to debug (e.g. pdb)
#     # device = torch.device("cuda:1")
#     # evaluate_single_sample(WorkArgs(problem_id=15, sample_idx=0, run_name=RUN_NAME, dataset=dataset, device=device, num_correct_trials=5))

#     # this doesn't work fully yet
#     # monkey_style_parallal_process_eval(problem_id, samples_range)
