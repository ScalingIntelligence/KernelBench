################################################################################
# Helpers for Dataset
################################################################################

import os
import random
import re
import hashlib

REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBench")


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



def check_id_matches_name(problem_id: int, problem_name: str) -> bool:
    """Check if the problem_id matches the ID in the problem_name.

    Args:
        problem_id: The problem ID to check
        problem_name: Path to the problem file

    Returns:
        bool: True if the ID matches the filename prefix

    Raises:
        ValueError: If filename doesn't follow the expected format
    """
    basename = os.path.basename(problem_name)
    parts = basename.split('_')

    if len(parts) < 2:
        raise ValueError(
            f"Problem filename '{basename}' doesn't follow expected format '<id>_<name>.py'"
        )

    try:
        file_id = int(parts[0])
    except ValueError:
        raise ValueError(
            f"Problem filename '{basename}' doesn't start with a numeric ID"
        )

    return problem_id == file_id


class KernelBenchDataset():
    """Dataset object for easy access to problems by IDs and iteration over problems.

    Args:
        dataset_name: Name of the dataset
        level: KernelBench level (1, 2, or 3)
        use_subset: Whether to use the subset_dataset instead of full dataset
        dataset: List of problem file paths for the full dataset
        subset_dataset: List of problem file paths for a subset
    """

    def __init__(
        self,
        dataset_name: str,
        level: int,
        use_subset: bool = False,
        dataset: list[str] = None,
        subset_dataset: list[str] = None
    ):
        if level not in [1, 2, 3]:
            raise ValueError(f"level must be 1, 2, or 3, got {level}")

        self.dataset_name = dataset_name
        self.level = level
        self.use_subset = use_subset

        # Avoid mutable default arguments
        if dataset is None:
            dataset = []
        if subset_dataset is None:
            subset_dataset = []

        if use_subset:
            self.problems = subset_dataset
        else:
            self.problems = dataset

    def get_problem_by_id(self, problem_id: int) -> str:
        """Get problem path by its ID (1-indexed logical index).

        Args:
            problem_id: The problem ID to search for

        Returns:
            str: Path to the problem file

        Raises:
            ValueError: If problem ID not found in dataset
        """
        for problem in self.problems:
            if check_id_matches_name(problem_id, problem):
                return problem
        raise ValueError(f"Problem ID {problem_id} not found in dataset")
    
    def get_problem_ids(self) -> list[int]:
        """Get list of all problem IDs in the dataset.

        Returns:
            list[int]: Sorted list of problem IDs extracted from filenames
        """
        return sorted([int(os.path.basename(problem).split('_')[0]) for problem in self.problems])

    def __len__(self) -> int:
        """Return the number of problems in the dataset."""
        return len(self.problems)

    def __getitem__(self, index: int) -> str:
        """Get problem by index (0-indexed, for backward compatibility).

        Args:
            index: Zero-based index into the problems list

        Returns:
            str: Path to the problem file
        """
        return self.problems[index]

    def __iter__(self):
        """Iterate over problem paths in the dataset."""
        return iter(self.problems)

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        subset_str = " (subset)" if self.use_subset else ""
        return (
            f"KernelBenchDataset(name='{self.dataset_name}', "
            f"level={self.level}, problems={len(self.problems)}{subset_str})"
        )


def fetch_ref_arch_from_dataset(
    dataset: "KernelBenchDataset",
    problem_id: int
) -> tuple[str, str, str]:
    """Fetch the reference architecture from the dataset.

    This is a shared utility function to avoid duplication across scripts.

    Args:
        dataset: KernelBenchDataset object
        problem_id: Logical index (1-indexed), matching the problem_id in the problem_name

    Returns:
        tuple containing:
            - ref_arch_path: Path to the reference architecture
            - ref_arch_name: Name of the reference architecture file
            - ref_arch_src: Source code of the reference architecture
    """
    from .utils import read_file

    ref_arch_path = dataset.get_problem_by_id(problem_id)
    ref_arch_src = read_file(ref_arch_path)
    ref_arch_name = os.path.basename(ref_arch_path)
    return (ref_arch_path, ref_arch_name, ref_arch_src)


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


def construct_kernelbench_dataset(level: int) -> KernelBenchDataset:
    dataset_list = construct_problem_dataset_from_problem_dir(
        os.path.join(KERNEL_BENCH_PATH, f"level{level}")
    )
    return KernelBenchDataset(
        dataset_name=f"KernelBench_Level_{level}",
        level=level,
        dataset=dataset_list
    )


KERNELBENCH_LEVEL_1_DATASET = construct_kernelbench_dataset(level=1)
KERNELBENCH_LEVEL_2_DATASET = construct_kernelbench_dataset(level=2)
KERNELBENCH_LEVEL_3_DATASET = construct_kernelbench_dataset(level=3)

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