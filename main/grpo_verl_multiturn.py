import torch
import os
import sys
from uuid import uuid4

from verl.interactions.base import BaseInteraction

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.run_utils import fetch_baseline_results, write_kernel_to_disk, write_eval_result_to_separate_file
from src.prompt_constructor import exec_result_to_exeution_feedback
from src.utils import extract_last_code

from main.evaluation_utils import send_evaluation_request, EvaluationWorkArgs


# RUNS_DIR = "/data/user_data/gyeongwk/KernelBench/grpo/runs"
RUNS_DIR = os.path.join(REPO_ROOT, "runs")
RUN_NAME = "grpo_train_multi_turn_Qwen2.5-7B-Instruct-SFT"
EVAL_SERVER_HOST = "shire-1-10"
EVAL_SERVER_PORT = 8085
NUM_GENERATIONS = 8
HARDWARE = "A6000_babel"

os.makedirs(os.path.join(RUNS_DIR, RUN_NAME), exist_ok=True)

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


def compute_score(data_source, solution_str, ground_truth, extra_info=None, thread_id=None, iteration=0):
    split, level, problem = extra_info['split'], extra_info['level'], extra_info['problem']
    run_dir = os.path.join(RUNS_DIR, RUN_NAME)

    sample_id = thread_id * 10 + iteration

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
        
        return reward_from_exec_result(level, problem, exec_result), exec_result

    print(f"No kernel src found for level {level} problem {problem} sample {sample_id}")
    return 0.0, None


# Define Interaction Environment for multi-turn support
class KernelBenchInteraction(BaseInteraction):
    def __init__(self, config):
        super().__init__(config)
        self._instance_dict = {}
        self.thread_id = {}
    
    async def start_interaction(self, instance_id=None, ground_truth=None, **kwargs):
        if instance_id is None:
            instance_id = str(uuid4())
        
        level = kwargs["level"]
        problem = kwargs["problem"]
        key = f"{level}_{problem}"
        if key not in self.thread_id:
            self.thread_id[key] = 0
        else:
            self.thread_id[key] += 1
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth,
            "response": None,
            "reward": 0.0,
            "thread_id": self.thread_id[key],
            "iteration": 0
        }
        print(f"Initializing interaction for level {level} problem {problem} with thread id {self.thread_id[key]}")
        return instance_id

    async def generate_response(self, instance_id, messages, **kwargs):
        content = ""
        for item in reversed(messages):
            if item.get("role") == "assistant":
                content = item.get("content", "")
                break
        
        self._instance_dict[instance_id]["response"] = content
        
        reward, exec_result = await self.run_kernel_evaluation(instance_id, kwargs)
        if exec_result is not None:
            message = exec_result_to_exeution_feedback(exec_result) 
        else:
            message = "Your kernel failed to execute. Please try again."
        self._instance_dict[instance_id]["iteration"] += 1
        return False, message, reward, {"thread_id": self._instance_dict[instance_id]["thread_id"]}
    
    async def run_kernel_evaluation(self, instance_id, extra_info):
        return compute_score("KernelBench", self._instance_dict[instance_id]["response"], self._instance_dict[instance_id]["ground_truth"], extra_info, self._instance_dict[instance_id]["thread_id"], self._instance_dict[instance_id]["iteration"])

    async def finalize_interaction(self, instance_id, **kwargs):
        del self._instance_dict[instance_id]


