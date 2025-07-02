################################################################################
# Helpers for Dataset
################################################################################

import os
import sys
import random
import re
import hashlib
import json
from datasets import Dataset
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from main.prompts import prompt_base
from src.utils import read_file

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")
RUNS_DIR = os.path.join(REPO_TOP_PATH, "runs")


def assign_problem_hash(problem_path: str) -> list[int]:
    """
    Assign a unique hash to a problem in the dataset
    """
    with open(problem_path, "r") as f:
        problem_src = f.read()
    return get_code_hash(problem_src)


def get_code_hash(problem_src: str) -> str:
    """
    Assign a unique hash to some piece of code
    Important to strip out the comments and whitespace as they are not functionally part of the code
    """
    # Remove multi-line comments first
    problem_src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', "", problem_src)
    # Remove inline comments and all whitespace
    cleaned_problem_src = re.sub(r"#.*$|\s+", "", problem_src, flags=re.MULTILINE)
    # hash only on code
    return hashlib.md5(cleaned_problem_src.encode()).hexdigest()


def construct_problem_dataset_from_problem_dir(problem_dir: str) -> list[str]:
    """
    Construct a list of relative paths to all the python files in the problem directory
    Sorted by the numerical prefix of the filenames
    """
    DATASET = []

    for file_name in os.listdir(problem_dir):
        if file_name.endswith(".py"):
            # TODO: revisit later to satisfy eval harnes
            relative_path = os.path.join(problem_dir, file_name)
            DATASET.append(relative_path)

    # Sort the DATASET based on the numerical prefix of the filenames
    DATASET.sort(key=lambda x: int(os.path.basename(x).split("_")[0]))

    return DATASET


def construct_kernelbench_dataset(level: int) -> list[str]:
    return construct_problem_dataset_from_problem_dir(
        os.path.join(KERNEL_BENCH_PATH, f"level{level}")
    )


KERNELBENCH_LEVEL_1_DATASET = construct_kernelbench_dataset(level=1)
KERNELBENCH_LEVEL_2_DATASET = construct_kernelbench_dataset(level=2)
KERNELBENCH_LEVEL_3_DATASET = construct_kernelbench_dataset(level=3)

# Define train and eval sets
TRAIN_PROBLEM_IDS_LEVEL_1 = [1, 4, 6, 8, 11, 12, 13, 15, 21, 22, 25, 27, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 46, 47, 49, 50, 52, 54, 56, 58, 59, 62, 63, 64, 65, 71, 72, 73, 74, 78, 79, 80, 81, 82, 83, 84, 85, 87, 91, 96]
KEVIN_TRAIN_PROBLEM_IDS_LEVEL_1 = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 31, 32, 33, 35, 36, 37, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
TRAIN_PROBLEM_IDS_LEVEL_1 = KEVIN_TRAIN_PROBLEM_IDS_LEVEL_1 # set to Kevin's train set
TEST_PROBLEM_IDS_LEVEL_1 = [i for i in range(1, 101) if i not in TRAIN_PROBLEM_IDS_LEVEL_1]

TRAIN_PROBLEM_IDS_LEVEL_2 = [1, 6, 7, 8, 11, 14, 15, 20, 25, 26, 27, 33, 36, 38, 42, 43, 44, 45, 47, 49, 51, 53, 55, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 79, 81, 84, 85, 88, 89, 90, 96, 98, 99]
KEVIN_TRAIN_PROBLEM_IDS_LEVEL_2 = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57, 60, 61, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
TRAIN_PROBLEM_IDS_LEVEL_2 = KEVIN_TRAIN_PROBLEM_IDS_LEVEL_2 # set to Kevin's train set
TEST_PROBLEM_IDS_LEVEL_2 = [i for i in range(1, 101) if i not in TRAIN_PROBLEM_IDS_LEVEL_2]


def check_in_train_dataset(level: int, problem_id: int) -> bool:
    if level == 1:
        return problem_id in TRAIN_PROBLEM_IDS_LEVEL_1
    elif level == 2:
        return problem_id in TRAIN_PROBLEM_IDS_LEVEL_2
    else:
        return False


################################################################################
# Fetch Reference Architecture
################################################################################
def fetch_ref_arch_from_problem_id(dataset, problem_id: int, dataset_src: str) -> tuple[str, str] | None:
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

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({problem_id})"
    
    return ref_arch_src, problem_name

def fetch_ref_arch_from_level_problem_id(level: int, problem_id: int, dataset_src: str) -> tuple[str, str] | None:
    if dataset_src == "local":
        directory = os.path.join(KERNEL_BENCH_PATH, f"level{level}")
        for file in os.listdir(directory):
            if file.startswith(f"{problem_id}_") and file.endswith(".py"):
                ref_arch_path = os.path.join(directory, file)
                ref_arch_src = read_file(ref_arch_path)
                problem_name = os.path.basename(ref_arch_path)
                return ref_arch_src, problem_name
        raise FileNotFoundError(f"No file found starting with '{problem_id}_' and ending with '.py' in {directory}")
    elif dataset_src == "huggingface":
        dataset = construct_kernelbench_dataset(level)
        return fetch_ref_arch_from_problem_id(dataset, problem_id, dataset_src)
    else:
        raise ValueError(f"Invalid dataset_src: {dataset_src}")



################################################################################
# Eval on Subsets of KernelBench
################################################################################


def get_kernelbench_subset(
    level: int, num_subset_problems: int = 10, random_seed: int = 42
) -> tuple[list[str], list[int]]:
    """
    Get a random subset of problems from the KernelBench dataset
    """

    full_dataset = construct_kernelbench_dataset(level)

    random.seed(random_seed)
    num_subset_problems = min(num_subset_problems, len(full_dataset))
    subset_indices = random.sample(range(len(full_dataset)), num_subset_problems)

    subset = sorted([full_dataset[i] for i in subset_indices])
    return subset, subset_indices


################################################################################
# Representative subsets of KernelBench
# use this if you want to iterate on methods without the hassle of running the full dataset
# problem_ids are 1-indexed (logical index)
################################################################################

level1_representative_subset = [
    "1_Square_matrix_multiplication_.py",
    "3_Batched_matrix_multiplication.py",
    "6_Matmul_with_large_K_dimension_.py",
    "18_Matmul_with_transposed_both.py",
    "23_Softmax.py",
    "26_GELU_.py",
    "33_BatchNorm.py",
    "36_RMSNorm_.py",
    "40_LayerNorm.py",
    "42_Max_Pooling_2D.py",
    "48_Mean_reduction_over_a_dimension.py",
    "54_conv_standard_3D__square_input__square_kernel.py",
    "57_conv_transposed_2D__square_input__square_kernel.py",
    "65_conv_transposed_2D__square_input__asymmetric_kernel.py",
    "77_conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__.py",
    "82_conv_depthwise_2D_square_input_square_kernel.py",
    "86_conv_depthwise_separable_2D.py",
    "87_conv_pointwise_2D.py",
]

level1_representative_subset_problem_ids = [1, 3, 6, 18, 23, 26, 33, 36, 40, 42, 48, 54, 57, 65, 77, 82, 86, 87]

level2_representative_subset = [
    "1_Conv2D_ReLU_BiasAdd.py",
    "2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide.py",
    "8_Conv3d_Divide_Max_GlobalAvgPool_BiasAdd_Sum.py",
    "18_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp.py",
    "23_Conv3d_GroupNorm_Mean.py",
    "28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply.py",
    "33_Gemm_Scale_BatchNorm.py",
    "43_Conv3d_Max_LogSumExp_ReLU.py",
]

level2_representative_subset_problem_ids = [1, 2, 8, 18, 23, 28, 33, 43]

level3_representative_subset = [
    "1_MLP.py",
    "5_AlexNet.py",
    "8_ResNetBasicBlock.py",
    "11_VGG16.py",
    "20_MobileNetV2.py",
    "21_EfficientNetMBConv.py",
    "33_VanillaRNN.py",
    "38_LTSMBidirectional.py",
    "43_MinGPTCausalAttention.py",
]

level3_representative_subset_problem_ids = [1, 5, 8, 11, 20, 33, 38, 43]


################################################################################
# Data set for verl GRPO
################################################################################
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


def search_for_best_kernels():
    k = 5
    for level in [1, 2]:
        best_k_kernels = {} # problem_id -> best kernels
        for directory in os.listdir(RUNS_DIR):
            if f"level{level}" in directory and "DeepSeek" in directory:
                print(f"Searching for best kernels in {directory}")
                eval_file_path = os.path.join(RUNS_DIR, directory, "eval_results.json")
                if not os.path.exists(eval_file_path):
                    print(f"No eval results found for {directory}")
                    continue
                with open(eval_file_path, "r") as f:
                    eval_results = json.load(f)
                for problem, samples in eval_results[f"{level}"].items():
                    if problem not in best_k_kernels:
                        best_k_kernels[problem] = []
                    for sample_id, eval_result in samples.items():
                        if eval_result["correctness"]: # initial filter for correct
                            eval_result["run_name"] = directory
                            best_k_kernels[problem].append(eval_result)
        # sort by runtime
        for problem, kernels in best_k_kernels.items():
            best_k_kernels[problem].sort(key=lambda x: x["runtime"])
            best_k_kernels[problem] = best_k_kernels[problem][:k]

        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "w") as f:
            json.dump(best_k_kernels, f, indent=2)
        print(f"Best {k} kernels for level {level} saved to {os.path.join(KERNEL_BENCH_PATH, f'best_k_kernels_level{level}.json')}")


def process_dataset_for_sft(k=1):
    sft_dataset = []
    sft_eval_dataset = []
    for level in [1, 2]:
        TRAIN_SET = TRAIN_PROBLEM_IDS_LEVEL_1 if level == 1 else TRAIN_PROBLEM_IDS_LEVEL_2
        with open(os.path.join(KERNEL_BENCH_PATH, f"best_k_kernels_level{level}.json"), "r") as f:
            best_k_kernels = json.load(f)
        for problem, eval_results in best_k_kernels.items():
            for eval_result in eval_results[:k]:
                run_name = eval_result["run_name"]
                kernel_path = os.path.join(RUNS_DIR, run_name, f"level_{level}_problem_{problem}_sample_{eval_result['sample_id']}_kernel.py")
                kernel_src = read_file(kernel_path)

                ref_arch_src, _ = fetch_ref_arch_from_level_problem_id(level, problem, "local")
                question = prompt_base(ref_arch_src)
                answer = kernel_src
                if int(problem) in TRAIN_SET:
                    sft_dataset.append((question, answer, level, problem))
                else:
                    sft_eval_dataset.append((question, answer, level, problem))
    
    df = Dataset.from_pandas(pd.DataFrame(sft_dataset, columns=["question", "answer", "level", "problem"]))
    df.to_parquet(os.path.join(KERNEL_BENCH_PATH, f"sft_dataset_best_{k}_train.parquet"))
    print(f"SFT dataset for level {level} saved to {os.path.join(KERNEL_BENCH_PATH, f'sft_dataset_best_{k}_train.parquet')}")
    df = Dataset.from_pandas(pd.DataFrame(sft_eval_dataset, columns=["question", "answer", "level", "problem"]))
    df.to_parquet(os.path.join(KERNEL_BENCH_PATH, f"sft_dataset_best_{k}_eval.parquet"))
    print(f"SFT eval dataset for level {level} saved to {os.path.join(KERNEL_BENCH_PATH, f'sft_dataset_best_{k}_eval.parquet')}")



if __name__ == "__main__":
    process_dataset_for_sft(k=1)
    process_dataset_for_sft(k=2)
    process_dataset_for_sft(k=3)
    process_dataset_for_sft(k=4)