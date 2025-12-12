'''
Example Usage:
python scripts/generate_and_eval_single_sample_modal.py dataset_src=huggingfac level=1 problem_id=1 eval_mode=modal gpu=L40S 
    server_type=deepseek model_name=deepseek-coder max_tokens=4096 temperature=0.0
'''

import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json
import modal

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor_toml import get_prompt_for_backend, get_custom_prompt
from src.utils import extract_first_code, extract_cuda_and_python_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

app = modal.App("eval_single_sample")

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_BENCH_PATH = os.path.join(REPO_TOP_DIR, "KernelBench")
SCRIPTS_PATH = os.path.join(REPO_TOP_DIR, "scripts")

torch.set_printoptions(precision=4, threshold=10)

gpu_arch_mapping = {"L40S": ["Ada"], "H100": ["Hopper"], "A100": ["Ampere"], "L4": ["Ada"], "T4": ["Turing"], "A10G": ["Ampere"]}

class EvalConfig(Config):
    def __init__(self):
        
        self.dataset_src = REQUIRED # either huggingface or local

        # name of dataset name on Hugging Face
        self.dataset_name = "ScalingIntelligence/KernelBench"


        # Problem Specification
        self.level = REQUIRED
        # NOTE: this is the logical index (problem id the problem_name)\
        self.problem_id = REQUIRED

        # Evaluation
        # local (requires a GPU), modal (cloud GPU) coming soon
        self.eval_mode = "modal"
        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu = "L40S"
        self.gpu_arch = ['Ada']
        self.precision = "fp32" # options ["fp32", "fp16", "bf16"]

        # Inference config
        self.server_type = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = None
        
        # Reasoning model specific parameters
        self.is_reasoning_model = False  # set to True for o1, o3, Gemini 2.5 thinking, etc.
        self.reasoning_effort = None  # for o1/o3: "low", "medium", "high"
        self.budget_tokens = 0  # for Claude extended thinking mode
        
        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

        self.backend = "cuda"
        # Prompt generation settings
        self.prompt_option = "one_shot"  # zero_shot, one_shot, few_shot
        self.include_hardware_info = False
        self.hardware_gpu_name = None
        self.custom_prompt_key = None

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

# ThunderKittens support - use TK image if directory exists locally
THUNDERKITTENS_LOCAL_PATH = os.path.join(REPO_TOP_DIR, "ThunderKittens")

SRC_PATH = os.path.join(REPO_TOP_DIR, "src")

if os.path.isdir(THUNDERKITTENS_LOCAL_PATH):
    # ThunderKittens image with TK environment and mounting
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
        .apt_install("git", "gcc-10", "g++-10", "clang")
        .pip_install_from_requirements(os.path.join(REPO_TOP_DIR, "requirements.txt"))
        .pip_install("pybind11")  # Ensure pybind11 is available for ThunderKittens compilation
        .env({
            "THUNDERKITTENS_ROOT": "/root/ThunderKittens",
            "THUNDERKITTENS_PATH": "/root/ThunderKittens",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "CXX": "g++-10",
            "CC": "gcc-10",
        })
        .add_local_dir(THUNDERKITTENS_LOCAL_PATH, remote_path="/root/ThunderKittens", copy=True)
        .add_local_dir(KERNEL_BENCH_PATH, remote_path="/root/KernelBench")
        .add_local_dir(SRC_PATH, remote_path="/root/src")
        .add_local_dir(SCRIPTS_PATH, remote_path="/root/scripts")
    )
else:
    # Standard image
    image = (
        modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
        .apt_install("git", "gcc-10", "g++-10", "clang")
        .pip_install_from_requirements(os.path.join(REPO_TOP_DIR, "requirements.txt"))
        .pip_install("pybind11")  # Ensure pybind11 is available
        .add_local_dir(KERNEL_BENCH_PATH, remote_path="/root/KernelBench")
        .add_local_dir(SRC_PATH, remote_path="/root/src")
        .add_local_dir(SCRIPTS_PATH, remote_path="/root/scripts")
    )


@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_kernel, verbose, gpu_arch, backend, precision, cuda_src=None, cuda_module_name="tk_kernels"):
        # 3. Evaluate Kernel
        # NOTE: no need to wrap around process here as only a single sample
        # see batch eval for examples of process isolation
        from src.eval import eval_kernel_against_ref
        from src.eval import get_torch_dtype_from_string
        # Use utility function to set the GPU architecture in the modal environment
        from src.utils import set_gpu_arch as modal_set_gpu_arch
        import sys
        import os
        
        # Add scripts directory to path for importing tk_compile
        if "/root/scripts" not in sys.path:
            sys.path.insert(0, "/root/scripts")
        from tk_compile import compile_cuda_on_modal
        
        modal_set_gpu_arch(gpu_arch)
        
        # If CUDA source provided, compile it first (for ThunderKittens)
        if cuda_src:
            cuda_module_path = compile_cuda_on_modal(cuda_src, cuda_module_name, gpu_arch, repo_top_path="/root")
            
            # Modify kernel_src to import the compiled module
            import_hook = f'''
import sys
import os
_tk_module_path = "{cuda_module_path}"
if _tk_module_path not in sys.path:
    sys.path.insert(0, _tk_module_path)
'''
            custom_kernel = import_hook + "\n" + custom_kernel
            print(f"[Modal] Modified kernel source to use compiled module at {cuda_module_path}")
        
        return eval_kernel_against_ref(
            ref_arch_src, custom_kernel, verbose=verbose, measure_performance=True, 
            num_correct_trials=5, num_perf_trials=100, backend=backend, precision=get_torch_dtype_from_string(precision)
        )

@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    
    """
    Keep it simple: Generate and evaluate a single sample
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
    
    print(f"Starting Eval with config: {config}")

    # Configurations
    
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.log:
        os.makedirs(config.logdir, exist_ok=True)
        
    # Problem Checks
    num_problems = len(curr_level_dataset)
    print(f"Number of problems in Level {config.level}: {num_problems}")
    print(f"Start Generation + Evaluation for Level {config.level} Problem {config.problem_id}")

    assert config.problem_id <= num_problems, f"Problem ID {config.problem_id} out of range for Level {config.level}"


    # 1. Fetch Problem
    if config.dataset_src == "huggingface":

        curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
        ref_arch_src = curr_problem_row["code"][0]
        problem_name = curr_problem_row["name"][0]

    elif config.dataset_src == "local":
        problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
        ref_arch_path = curr_level_dataset[problem_idx_in_dataset]

        problem_name = os.path.basename(ref_arch_path)
        ref_arch_src = read_file(ref_arch_path)
    # import pdb; pdb.set_trace()

    # Extract problem number from problem name (e.g. "1" from "1_Square_matrix_multiplication_.py")
    problem_number = int(problem_name.split("_")[0])
    assert problem_number == config.problem_id, f"Problem number in filename ({problem_number}) does not match config problem_id ({config.problem_id})"
    
    
    # 2. Generate Sample
    # Create inference function with config parameters
    # We provide some presets in utils but you can also pass in your own, see query_server for more details
    inference_server = create_inference_server_from_presets(server_type=config.server_type,
                                                        model_name=config.model_name,
                                                        temperature=config.temperature,
                                                        max_tokens=config.max_tokens,
                                                        verbose=config.verbose, 
                                                        time_generation=True,
                                                        is_reasoning_model=config.is_reasoning_model,
                                                        reasoning_effort=config.reasoning_effort,
                                                        budget_tokens=config.budget_tokens)
    

    custom_prompt_key = getattr(config, "custom_prompt_key", None)
    if isinstance(custom_prompt_key, str):
        trimmed = custom_prompt_key.strip()
        if trimmed.lower() in {"", "none"}:
            custom_prompt_key = None
        else:
            custom_prompt_key = trimmed
    config.custom_prompt_key = custom_prompt_key

    # Checks if user has inputted a valid argument for how many examples they want to give as context to the model
    prompt_option = str(config.prompt_option).lower()
    valid_prompt_options = {"zero_shot", "one_shot", "few_shot"}
    include_hardware = config.include_hardware_info
    if isinstance(include_hardware, str):
        include_hardware = include_hardware.lower() in ["true", "1", "yes"]
    config.include_hardware_info = include_hardware

    supported_backends = {"cuda", "triton", "tilelang", "cute", "thunderkittens"}
    backend = config.backend.lower()
    if backend not in supported_backends:
        raise ValueError(
            f"Unsupported backend: {config.backend}. Must be one of {sorted(supported_backends)}."
        )
    
    # ThunderKittens uses fp32 by default
    if backend == "thunderkittens":
        config.precision = "fp32"

    #tilelang only supports fp16 or bf16
    if backend == "tilelang":
        config.precision = "fp16"
        config.hardware_gpu_name = config.hardware_gpu_name or getattr(config, "gpu", None)

    if not custom_prompt_key:
        if prompt_option not in valid_prompt_options:
            raise ValueError(
                f"Invalid prompt_option '{config.prompt_option}'. Must be one of {sorted(valid_prompt_options)}."
            )
        if include_hardware and not config.hardware_gpu_name:
            raise ValueError(
                "include_hardware_info is True but hardware_gpu_name is not provided."
            )

    if custom_prompt_key:
        custom_prompt = get_custom_prompt(
            custom_prompt_key,
            ref_arch_src=ref_arch_src,
            backend=backend,
            option=prompt_option,
            precision=config.precision,
            include_hardware=include_hardware,
            gpu_name=config.hardware_gpu_name,
        )
    else:
        custom_prompt = get_prompt_for_backend(
            ref_arch_src,
            backend,
            option=prompt_option,
            precision=config.precision,
            include_hardware=include_hardware,
            gpu_name=config.hardware_gpu_name,
        )
        
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_prompt)

    # Query server with constructed prompt
    custom_kernel_response = inference_server(custom_prompt)
    
    # For ThunderKittens, extract both CUDA and Python code
    cuda_src = None
    if backend == "thunderkittens":
        cuda_code, python_code = extract_cuda_and_python_code(custom_kernel_response)
        if cuda_code is None or python_code is None:
            # Fallback to single code extraction
            print("[WARNING] Could not extract separate CUDA and Python code blocks, falling back to single extraction")
            custom_kernel = extract_first_code(custom_kernel_response, ["python", "cpp"])
            assert custom_kernel is not None, f"Custom {config.backend} kernel code generation failed"
        else:
            custom_kernel = python_code
            cuda_src = cuda_code
            print(f"[INFO] Extracted CUDA code ({len(cuda_src)} chars) and Python code ({len(custom_kernel)} chars)")
    else:
        custom_kernel = extract_first_code(custom_kernel_response, ["python", "cpp"])
        # check LLM is able to generate custom kernel code
        assert custom_kernel is not None, f"Custom {config.backend} kernel code generation failed"
    
    # Log generated files
    if config.log:
        if cuda_src:
            with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.cu"), "w") as f:
                f.write(cuda_src)
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_kernel)

    with app.run():
        kernel_exec_result = EvalFunc.with_options(gpu=config.gpu)().eval_single_sample_modal.remote(
            ref_arch_src, custom_kernel, config.verbose, gpu_arch_mapping[config.gpu], config.backend, config.precision, cuda_src=cuda_src, cuda_module_name="tk_kernels"
        )
        
        print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")
        
        if config.log:
            with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
                f.write(f"Problem Name: {problem_name}\n")
                f.write(str(kernel_exec_result))

if __name__ == "__main__":
    main()