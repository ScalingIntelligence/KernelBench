import json
import os, sys
from dataclasses import dataclass

import pydra
import torch

from datasets import load_dataset
from pydra import Config, REQUIRED

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt
from src.utils import (
    create_inference_server_from_presets,
    extract_first_code,
    maybe_multithread,
    read_file,
    set_gpu_arch,
)

"""
Batch Generate Samples for Particular Level

Assume 1 sample per problem here
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)


class GenerationConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED  # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED

        # subset of problems to generate, otherwise generate on all problems in the level
        self.subset = (
            None,
            None,
        )  # (problem_id, problem_name), these are the logical index

        self.run_name = REQUIRED  # name of the run

        # num of thread pool to call inference server in parallel
        self.num_workers = 64
        self.api_query_interval = 0.0

        # Inference config
        self.server_type = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = 0.0
        
        # Reasoning model specific parameters
        self.is_reasoning_model = False  # set to True for o1, o3, Gemini 2.5 thinking, etc.
        self.reasoning_effort = "low"  # for o1/o3: "low", "medium", "high"
        self.budget_tokens = 0  # for Claude extended thinking mode

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        self.verbose = False
        self.store_type = "local"  # TODO: add Database Integration

        # Number of samples to generate per problem for pass@k analysis
        self.num_samples = 1  # Default to 1 sample per problem

        self.log_prompt = False

        self.backend = "cuda"
        
        self.precision = "fp32"
        self.prompt_option = "one_shot"  # zero_shot, one_shot, few_shot
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None

    def greedy(self):
        # For greedy decoding, epsecially baseline eval
        self.greedy_sample = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@dataclass
class WorkArgs:
    problem_id: int  # logically indexed
    sample_id: int


def generate_sample_single(
    work: WorkArgs,
    config: GenerationConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
) -> bool:
    # 1. Fetch Problem
    if config.dataset_src == "huggingface":
        curr_problem_row = dataset.filter(
            lambda x: x["problem_id"] == work.problem_id, desc=None
        )

        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = (
            work.problem_id - 1
        )  # due to dataset list being 0-indexed locally
        ref_arch_path = dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert (
        problem_number == work.problem_id
    ), f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"

    if config.custom_prompt_key:
        custom_prompt = get_custom_prompt(
            config.custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    else:
        custom_prompt = get_prompt_for_backend(
            ref_arch_src,
            config.backend,
            option=config.prompt_option,
            precision=config.precision,
            include_hardware=config.include_hardware_info,
            gpu_name=config.hardware_gpu_name,
        )
    if config.log_prompt:
        prompt_path = os.path.join(
            run_dir,
            f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_prompt.txt",
        )
        with open(prompt_path, "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt
    custom_kernel = inference_server(custom_prompt)
    custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_kernel is not None, "Custom CUDA code generation failed"

    if config.verbose:
        print(
            f"Generated sample {work.sample_id} for problem {problem_number}: {problem_name}"
        )

    # Store to local file
    kernel_path = os.path.join(
        run_dir,
        f"level_{config.level}_problem_{work.problem_id}_sample_{work.sample_id}_kernel.py",
    )
    with open(kernel_path, "w") as f:
        f.write(custom_kernel)

    return True


def generate_sample_launcher(
    work: WorkArgs,
    config: GenerationConfig,
    dataset,
    inference_server: callable,
    run_dir: str,
):
    try:
        return generate_sample_single(work, config, dataset, inference_server, run_dir)
    except Exception as e:
        print(f"Error generating sample {work.problem_id} {work.sample_id}: {e}")
        return None


def check_kernel_exists(
    run_dir: str, level: int, problem_id: int, sample_id: int
) -> bool:
    """
    Check if a kernel for a given problem and sample ID already exists in the run directory
    """
    kernel_path = os.path.join(
        run_dir, f"level_{level}_problem_{problem_id}_sample_{sample_id}_kernel.py"
    )
    return os.path.exists(kernel_path)


@pydra.main(base=GenerationConfig)
def main(config: GenerationConfig):
    """
    Batch Generate Samples for Particular Level
    Store generated kernels in the specified run directory
    """
    from src.utils import SERVER_PRESETS
    
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if config.model_name is None or config.model_name == "None":
            config.model_name = preset.get("model_name", "None")
        if config.max_tokens is None or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", "None")
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", "None")
    
    # Convert string boolean to actual boolean for reasoning model flag
    if isinstance(config.is_reasoning_model, str):
        config.is_reasoning_model = config.is_reasoning_model.lower() in ['true', '1', 'yes']
    
    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        if trimmed.lower() in {"", "none"}:
            custom_prompt_key = None
        else:
            custom_prompt_key = trimmed
    config.custom_prompt_key = custom_prompt_key

    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware

    supported_backends = {"cuda", "triton", "cute", "tilelang"}
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be one of {sorted(supported_backends)}."
        )
    config.backend = backend
    if backend == "tilelang":
        config.precision = "fp16"

    config.prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot"}
    if not config.custom_prompt_key:
        if config.prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. Must be one of {sorted(valid_prompt_options)}."
            )
        if include_hardware and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

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
        assert (
            config.subset[0] >= 1 and config.subset[1] <= num_problems_in_level
        ), f"Subset range {config.subset} out of range for Level {config.level}"
        problem_id_range = range(config.subset[0], config.subset[1])

    print(
        f"Generating {config.num_samples} sample(s) each for level {config.level} problems: {problem_id_range}"
    )

    # set up run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    run_exists = os.path.exists(run_dir)
    if run_exists:
        print(f"\n⚠️  WARNING: Run directory already exists: {run_dir}")
        print(f"   Existing kernels will be skipped. Use a different run_name for a fresh run.\n")
    os.makedirs(run_dir, exist_ok=True)
    pydra.save_yaml(config.to_dict(), os.path.join(run_dir, "generation_config.yaml"))

    assert (
        config.store_type == "local"
    ), "supporting local file-system based storage for now"  # database integreation coming soon, need to migrate from CUDA Monkeys code

    problems_to_run = []
    total_problems = 0
    already_completed = 0
    for problem_id in range(
        problem_id_range.start, problem_id_range.stop + 1
    ):  # end index is inclusive
        for sample_id in range(config.num_samples):
            total_problems += 1
            if not check_kernel_exists(run_dir, config.level, problem_id, sample_id):
                problems_to_run.append(
                    WorkArgs(problem_id=int(problem_id), sample_id=sample_id)
                )
            else:
                already_completed += 1
    
    if already_completed > 0:
        print(f"📁 Found {already_completed}/{total_problems} kernels already generated. Generating remaining {len(problems_to_run)} kernels.")

    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
        is_reasoning_model=config.is_reasoning_model,
        reasoning_effort=config.reasoning_effort,
        budget_tokens=config.budget_tokens,
    )

    # Launch workers
    generation_results = maybe_multithread(
        generate_sample_launcher,
        problems_to_run,
        config.num_workers,
        time_interval=config.api_query_interval,
        # extra args
        config=config,
        dataset=curr_level_dataset,
        inference_server=inference_server,
        run_dir=run_dir,
    )

    num_generated_samples = len(generation_results)
    num_attempted = len(problems_to_run)
    num_failed_problems = num_attempted - num_generated_samples
    
    if num_attempted == 0:
        print(f"\n✅ All {total_problems} kernels already exist in {run_dir}")
        print(f"   Use a different run_name if you want to generate fresh samples.\n")
    else:
        print(
            f"\nGenerated {num_generated_samples} samples for total {num_attempted} problems, Please retry for the {num_failed_problems} failed problems."
        )


if __name__ == "__main__":
    main()
