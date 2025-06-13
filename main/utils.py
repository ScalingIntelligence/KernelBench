import os
import json

from src.utils import read_file



def check_if_kernel_exists(run_dir: str, level: int, problem_id: int, sample_id: int) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    return os.path.exists(kernel_path) 


def check_if_eval_exists_local(problem_id: int, sample_id: int, eval_file_path: str) -> bool:
    """
    Check if evaluation result already exists in eval results file
    """
    if os.path.exists(eval_file_path):
        with open(eval_file_path, 'r') as f:
            eval_results = json.load(f)
        return str(problem_id) in eval_results and str(sample_id) in eval_results[str(problem_id)]
    return False



def fetch_ref_arch_from_problem_id(dataset, problem_id: int, dataset_src: str) -> str | None:
    """
    Fetch reference architecture from problem directory
    Either from Hugging Face or Local Dataset
    """
    if dataset_src == "huggingface":
        curr_problem_row = dataset.filter(lambda x: x["problem_id"] == problem_id, num_proc=1, desc=None)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]
    
    elif dataset_src == "local":
        problem_idx_in_dataset = problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # verify
        # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    
    return ref_arch_src


def fetch_kernel_from_disk(run_dir: str, level: int, problem_id: int, sample_id: int) -> str | None:
    """
    Fetch kernel file from disk (stored in runs/{run_name})
    """
    kernel_path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py")
    
    if os.path.exists(kernel_path):
        return read_file(kernel_path)
    else:
        return None
