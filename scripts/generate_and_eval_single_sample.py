import pydra
from pydra import REQUIRED, Config
import os, sys
import torch
import json

from datasets import load_dataset

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import extract_first_code, query_server, set_gpu_arch, read_file, create_inference_server_from_presets

"""
Generate and evaluate a single sample
Easiest way to get started, to test a single problem for experimentation or debugging
"""

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

torch.set_printoptions(precision=4, threshold=10)

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
        self.eval_mode = "local"
        # GPU Architecture target(s) for compilation.
        # Previously hard-coded to ["Ada"], which caused kernels to be built for SM89 on older GPUs (e.g. SM75 Turing),
        # producing: CUDA error: no kernel image is available for execution on the device.
        # We now auto-detect the current device's compute capability and map it to a friendly name.
        # You can still override via CLI: gpu_arch=["75"] or gpu_arch=["Turing"].
        self.gpu_arch = None  # defer detection until main()

        # Inference config
        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
        self.max_tokens = 4096
        self.temperature = 0.0

        # Logging
        self.logdir = os.path.join(REPO_TOP_DIR, "results/eval_logs")
        self.verbose = False

        self.log = False
        self.log_prompt = False
        self.log_generated_kernel = False
        self.log_eval_result = False

    def verbose_logging(self):
        self.log = True
        self.log_prompt = True
        self.log_generated_kernel = True
        self.log_eval_result = True

    def __repr__(self):
        return f"EvalConfig({self.to_dict()})"


@pydra.main(base=EvalConfig)
def main(config: EvalConfig):
    """
    Keep it simple: Generate and evaluate a single sample
    """
    print(f"Starting Eval with config: {config}")

    # Configurations

    # --- GPU Arch Auto-Detection -------------------------------------------------
    # If user did not supply gpu_arch, detect current device and set both config.gpu_arch
    # (for logging / reproducibility) and export TORCH_CUDA_ARCH_LIST so torch cpp_extension
    # produces SASS+PTX for the correct SM. If the user already exported TORCH_CUDA_ARCH_LIST
    # we preserve it unless it mismatches the detected arch (warn but keep user's setting).
    if config.gpu_arch in (None, [], [None]):
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            major, minor = cc
            sm_str = f"{major}{minor}"
            # Friendly name mapping (extend as needed)
            sm_name_map = {
                "75": "Turing",  # e.g. GTX 16xx / RTX 20xx
                "80": "Ampere",  # A100 (SM80)
                "86": "Ampere",  # RTX 30xx GA10x
                "89": "Ada",     # L40S / some Ada datacenter
                "90": "Hopper",  # H100
                "102": "Blackwell"  # example future mapping
            }
            friendly = sm_name_map.get(sm_str, f"SM{sm_str}")
            config.gpu_arch = [friendly]

            env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
            desired_arch_list = sm_str
            if env_arch_list is None:
                os.environ["TORCH_CUDA_ARCH_LIST"] = desired_arch_list
                if config.verbose:
                    print(f"[GPU ARCH] Auto-set TORCH_CUDA_ARCH_LIST={desired_arch_list} ({friendly})")
            else:
                # check if current sm present; normalize to handle forms like '7.5' vs '75'
                normalized_env = env_arch_list.replace(".", "")
                if sm_str not in normalized_env:
                    print(f"[GPU ARCH][WARNING] Detected SM {sm_str} ({friendly}) not in existing TORCH_CUDA_ARCH_LIST='{env_arch_list}'.\n"
                          f"Compilation may target the wrong architecture. Consider: export TORCH_CUDA_ARCH_LIST={sm_str}")
        else:
            print("[GPU ARCH][WARNING] CUDA not available; proceeding without setting gpu_arch")

    # If still no gpu_arch (e.g., CPU only) leave unset so build uses default (all arch PTX) or fails fast.

    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    if config.gpu_arch:
        # Accept either friendly names or raw SM numbers
        set_gpu_arch(config.gpu_arch)  # sets TORCH_CUDA_ARCH_LIST if provided as list of names / numbers

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
                                                        time_generation=True)
    


    custom_cuda_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src)
    if config.log_prompt:
        with open(os.path.join(config.logdir, f"prompt_level_{config.level}_problem_{config.problem_id}.txt"), "w") as f:
            f.write(custom_cuda_prompt)

    # Query server with constructed prompt
    custom_cuda = inference_server(custom_cuda_prompt)
    custom_cuda = extract_first_code(custom_cuda, ["python", "cpp"])
    # check LLM is able to generate custom CUDA code
    assert custom_cuda is not None, "Custom CUDA code generation failed"
    
    # this should be optional
    if config.log:
        with open(os.path.join(config.logdir, f"generated_kernel_level_{config.level}_problem_{config.problem_id}.py"), "w") as f:
            f.write(custom_cuda)

    # 3. Evaluate Kernel
    # NOTE: no need to wrap around process here as only a single sample
    # see batch eval for examples of process isolation
    kernel_exec_result = eval_kernel_against_ref(
        ref_arch_src, custom_cuda, verbose=config.verbose, measure_performance=True, num_correct_trials=5, num_perf_trials=100
    )
    
    print(f"Evaluation result for level {config.level} problem {config.problem_id}:\n{kernel_exec_result}")

    if config.log:
        with open(os.path.join(config.logdir, f"eval_result_level_{config.level}_problem_{config.problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))


if __name__ == "__main__":
    main()

