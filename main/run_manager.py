"""
Manages the directory created during runs.
- Create a run directory
- Save config
- fetch kernels
- fetch eval results

"""
import os
import json
 
from src.utils import read_file

################################################################################
# Kernel Fetching
################################################################################
def check_if_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    return os.path.exists(kernel_path) 


def find_highest_sample_id(run_dir: str, level: int, problem_id: int) -> int:
    """
    Find the highest sample ID for a given problem
    """
    sample_ids = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(run_dir) if f.startswith(f"level_{level}_problem_{problem_id}_sample_")]
    return max(sample_ids)


def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> tuple[str, str] | None:
    """
    Fetch kernel file from disk (stored in runs/{run_name})
    """
    kernel_name = f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    kernel_path = os.path.join(run_dir, kernel_name)
    
    if os.path.exists(kernel_path):
        return read_file(kernel_path), kernel_name
    else:
        return None, None


################################################################################
# Evaluation Fetching
################################################################################
def check_if_eval_exists_local(level: int, problem_id: int, sample_id: int, eval_file_path: str) -> bool:
    """
    Check if evaluation result already exists in eval results file
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
        return str(level) in eval_results and str(problem_id) in eval_results[str(level)] and str(sample_id) in eval_results[str(level)][str(problem_id)]
    return False


def fetch_eval_results_for_problem(level: int, problem_id: int, eval_file_path: str) -> list[str]:
    """
    Get evaluated kernels from disk
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
        return eval_results[str(level)][str(problem_id)]
    return {}


def fetch_eval_result_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> dict | None:
    """
    Fetch evaluation result from disk (stored in runs/{run_name})
    """
    eval_path = os.path.join(run_dir, f"eval_results.json")
    
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            eval_results = json.load(f)
        if str(level) in eval_results and str(problem_id) in eval_results[str(level)] and str(sample_id) in eval_results[str(level)][str(problem_id)]:
            return eval_results[str(level)][str(problem_id)][str(sample_id)]
    return None

