import torch
import time
import os
import sys
import yaml
import pandas as pd
import json
from datasets import Dataset

from verl.interactions.base import BaseInteraction
from uuid import uuid4
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from main.prompts import prompt_base
from main.evaluation_utils import send_evaluation_request, EvaluationWorkArgs
from main.dataset import fetch_ref_arch_from_level_problem_id, TRAIN_PROBLEM_IDS_LEVEL_1, TRAIN_PROBLEM_IDS_LEVEL_2, KERNEL_BENCH_PATH
from main.run_utils import find_highest_sample_id, fetch_baseline_results, write_kernel_to_disk, write_eval_result_to_separate_file

from src.utils import set_gpu_arch, extract_last_code

RUNS_DIR = os.path.join(REPO_ROOT, "runs")
RUN_NAME = "grpo_verl_test"
EVAL_SERVER_HOST = "babel-7-17"
EVAL_SERVER_PORT = 8083
NUM_GENERATIONS = 8
HARDWARE = "A6000_babel"

os.makedirs(os.path.join(RUNS_DIR, RUN_NAME), exist_ok=True)

# Dataset conversion
def get_train_dataset():
    return [(1, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_1] + [(2, problem) for problem in TRAIN_PROBLEM_IDS_LEVEL_2] # for now use level 1 for training

def get_eval_dataset():
    return [(1, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_1] + [(2, problem) for problem in range(1, 101) if problem not in TRAIN_PROBLEM_IDS_LEVEL_2] # for now use level 2 for evaluation
 

def construct_dataset(train=True):
    if train:
        dataset = get_train_dataset()
    else:
        dataset = get_eval_dataset()

    qa_dataset = []
    for (level, problem) in dataset:
        ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, "local")
        question = prompt_base(ref_arch_src)
        answer = ref_arch_src
        qa_dataset.append((question, answer, level, problem))
    
    df = Dataset.from_pandas(pd.DataFrame(qa_dataset, columns=["question", "answer", "level", "problem"]))
    return df

def make_map_fn(split):
    def process_fn(example):
        question = example.pop('question')

        answer = example.pop('answer')
        level = example.pop('level')
        problem = example.pop('problem')
        data = {
            "data_source": "KernelBench",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "reward_model": {
                "style": "custom",
                "ground_truth": answer 
            },
            "extra_info": {
                'split': split,
                'level': level,
                'problem': problem
            }
        }
        return data
    return process_fn


def process_dataset():
    train_dataset = construct_dataset(train=True)
    eval_dataset = construct_dataset(train=False)
    train_dataset = train_dataset.map(make_map_fn('train'))
    eval_dataset = eval_dataset.map(make_map_fn('eval'))

    train_dataset.to_parquet(os.path.join(KERNEL_BENCH_PATH, "train_dataset.parquet"))
    eval_dataset.to_parquet(os.path.join(KERNEL_BENCH_PATH, "eval_dataset.parquet"))


# Define custom reward functions
def reward_from_exec_result(level, problem, exec_result):
    if exec_result.correctness:
        try:
            baseline_results = fetch_baseline_results(level, problem, HARDWARE)
            speedup = baseline_results["mean"] / exec_result.runtime
            return 0.3 + float(speedup)
        except Exception as e:
            print(f"Error fetching baseline results for level {level} problem {problem}: {e}")
            return 0.3
    else:
        return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, i=None):
    split, level, problem = extra_info['split'], extra_info['level'], extra_info['problem']
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    thread_id = i # for now
    if split == "train":
        sample_id = find_highest_sample_id(run_dir, level, problem, thread_id, NUM_GENERATIONS) # batch_size
    else:
        sample_id = find_highest_sample_id(run_dir, level, problem, 0, 1) # just find the next sample_id


    response_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_response.txt")
    with open(response_path, "w") as f:
        f.write(solution_str)

    kernel_src = extract_last_code(solution_str, ["python", "cpp"])
    kernel_name = f"level_{level}_problem_{problem}_sample_{sample_id}"

    if kernel_src is not None:
        write_kernel_to_disk(run_dir, level, problem, sample_id, kernel_src)

    work_args=EvaluationWorkArgs(level=level, problem_id=problem, sample_id=sample_id, device=torch.device("cuda"))
    exec_result = send_evaluation_request(EVAL_SERVER_HOST, EVAL_SERVER_PORT, work_args, RUN_NAME, kernel_src, kernel_name)
    write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir)
    return reward_from_exec_result(level, problem, exec_result)




# Define Interaction Environment for multi-turn support
class KernelBenchInteraction(BaseInteraction):
    def __init__(self, config):
        super().__init__(config)
        self._instance_dict = {}
    
    async def start_interaction(self, instance_id=None, ground_truth=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "response": None,
            "reward": 0.0,
        }
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        content = ""
        for item in reversed(messages):
            if item.get("role") == "user":
                content = item.get("content", "")
                break
        
        self._instance_dict[instance_id]["response"] = content
        
        reward, message = await self.run_kernel_evaluation(instance_id)
        return False, message, reward, {}
    
    async def run_kernel_evaluation(self, instance_id):
        return compute_score("KernelBench", self._instance_dict[instance_id]["response"], self._instance_dict[instance_id]["ground_truth"], self._instance_dict[instance_id]["extra_info"])

    async def finalize_interaction(self, instance_id, **kwargs):
        del self._instance_dict[instance_id]



if __name__ == "__main__":
    process_dataset()