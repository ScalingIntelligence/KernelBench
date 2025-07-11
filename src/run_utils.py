"""
Manages the directory created during runs.
- Create a run directory
- Save config
- fetch kernels
- fetch eval results

"""
import os
import json

from .utils import read_file
from .eval import check_metadata_serializable_all_types

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

################################################################################
# Kernel Fetching
################################################################################
def to_kernel_name(level: int, problem_id: int, sample_id: int) -> str:
    return f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"

def check_if_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(run_dir, to_kernel_name(level, problem_id, sample_id))
    return os.path.exists(kernel_path) 


def find_highest_sample_id(run_dir: str, level: int, problem_id: int, thread_id: int, batch_size: int) -> int:
    """
    Find the highest sample ID for a given problem starting at thread_id and incrementing by batch_size
    """
    sample_id = thread_id
    while check_if_kernel_exists(run_dir, level, problem_id, sample_id):
        sample_id += batch_size
    return sample_id


def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> tuple[str, str] | None:
    """
    Fetch kernel file from disk (stored in runs/{run_name})
    """
    kernel_name = to_kernel_name(level, problem_id, sample_id)
    kernel_path = os.path.join(run_dir, kernel_name)
    
    if os.path.exists(kernel_path):
        return read_file(kernel_path), kernel_name
    else:
        return None, None

def write_kernel_to_disk(run_dir: str, level: int, problem_id: int, sample_id: int, kernel_src: str):
    kernel_name = to_kernel_name(level, problem_id, sample_id)
    kernel_path = os.path.join(run_dir, kernel_name)
    with open(kernel_path, 'w') as f:
        f.write(kernel_src)


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


def fetch_baseline_results(level: int, problem_id: int, hardware: str):
    baseline_path = os.path.join(REPO_TOP_DIR, "results", "timing", hardware, "baseline_time_torch.json")
    with open(baseline_path, 'r') as f:
        baseline_results = json.load(f)[f"level{str(level)}"]
    
    for prob_name, results in baseline_results.items():
        if int(prob_name.split("_")[0]) == problem_id:
            return results
    return None


def write_eval_result_to_separate_file(level, problem, sample_id, exec_result, run_dir):
    eval_result_path = os.path.join(run_dir, f"level_{level}_problem_{problem}_sample_{sample_id}_eval_result.json")
    eval_result = {
        'level': level,
        'problem_id': problem,
        'sample_id': sample_id,
        'compiled': exec_result.compiled,
        'correctness': exec_result.correctness,
        'metadata': check_metadata_serializable_all_types(exec_result.metadata),
        'runtime': exec_result.runtime,
        'runtime_stats': exec_result.runtime_stats,
    }

    with open(eval_result_path, "w") as f:
        json.dump(eval_result, f)


################################################################################
# Deprecated
################################################################################
# def update_eval_results(run_dir):
#     config_path = os.path.join(run_dir, "config.yaml")
#     if not os.path.exists(config_path):
#         print("No config file found. Using empty dict")
#         config = argparse.Namespace()
#     else:
#         with open(config_path, "r") as f:
#             config = yaml.load(f, Loader=yaml.FullLoader)

#     eval_file_path = os.path.join(run_dir, "eval_results.json")
#     with open(eval_file_path, "r") as f:
#         eval_results = json.load(f)

#     deprecated = False
#     for level, problems in eval_results.items():
#         for problem, samples in problems.items():
#             if "correctness" in samples:
#                 deprecated = True
#                 break
#             break
#         break

#     if deprecated:
#         print("Deprecated eval results found. Updating to new format")
#         eval_results = {f"{config['level']}": eval_results}
#         with open(eval_file_path, "w") as f:
#             json.dump(eval_results, f, indent=4)


# def update_eval_results_for_all_runs():
#     for directory in os.listdir(RUNS_DIR):
#         if "level" in directory and "DeepSeek" in directory:
#             update_eval_results(os.path.join(RUNS_DIR, directory))

