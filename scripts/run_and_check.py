import shutil
import torch
import pydra
from pydra import REQUIRED, Config
from dataclasses import dataclass
import os

from KernelBenchInternal.src import eval as kernel_eval
from KernelBenchInternal.src import utils as kernel_utils
from KernelBenchInternal.scripts import generate_baseline_time

from KernelBenchInternal.src.utils import read_file

"""
Run a single test and check if it compiles and runs correctly
Either do this from a file or from a database

1. Option 1: Ref: Manual Path, Solution: Manual path
2. Option 2: Ref: KernelBench Problem, Solution: Manual path
3. Option 3: Ref + Sample Fetch from database for solution and sample
"""

torch.set_printoptions(precision=4, threshold=10)

class ScriptConfig(Config):
    def __init__(self):
        self.run_name = "run_and_check" 
        # Input src definition
        self.eval_option = REQUIRED

        # option 1: 
        self.ref_arch_src_path = ""
        self.kernel_src_path = ""

        # option 2: 
        self.problem_id = ""

        # KernelBench Eval specific
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 300
        self.verbose = False

        # just running on one device
        self.measure_performance = True


        self.build_dir_prefix = ""

        self.only_evaluate_cached = False
        self.clear_cache = False

        self.gpu_arch = ["Ada"] # build for L40s Ada Lovelace architecture

    def __repr__(self):
        return f"ScriptConfig({self.to_dict()})"

def evaluate_single_sample_src(ref_arch_src: str, kernel_src: str, configs: dict, device: torch.device) -> kernel_eval.KernelExecResult:
    # fetch kernel code from database

    kernel_hash = str(hash(kernel_src))
    build_dir = os.path.join(configs["build_dir_prefix"], "test_build", kernel_hash)
    
    num_correct_trials = configs["num_correct_trials"]
    num_perf_trials = configs["num_perf_trials"]    
    verbose = configs["verbose"]
    measure_performance = configs["measure_performance"]
    try:
        eval_result = kernel_eval.eval_kernel_against_ref(
        original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance=measure_performance,
            verbose=verbose,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            build_dir=build_dir,
            device=device
        )
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result


@pydra.main(base=ScriptConfig)
def main(config: ScriptConfig):

    print("Running with config", config)
    
    '''
    Simple check
    '''

    assert config.eval_option == 1, "Only support option 1 for now"

    print("Using option 1: Ref: Manual Path, Solution: Manual path")
    assert config.ref_arch_src_path != "", "ref_arch_src_path is required"
    assert config.kernel_src_path != "", "kernel_src_path is required"  
    ref_arch_src = read_file(config.ref_arch_src_path)
    kernel_src = read_file(config.kernel_src_path)

    device = torch.device("cuda:0") # default device

    kernel_utils.set_gpu_arch(config.gpu_arch)

    kernel_eval_result = evaluate_single_sample_src(ref_arch_src, kernel_src, config.to_dict(), device)

    baseline_result = generate_baseline_time.measure_program_time(ref_arch_name=config.run_name, ref_arch_src=ref_arch_src, device=device)

    print(f"Kernel eval result: {kernel_eval_result}")
    print(f"Baseline result: {baseline_result}")
    
    if kernel_eval_result.correctness:
        print("Speedup: ", baseline_result["mean"] / kernel_eval_result.runtime)

    else:
        print("Kernel did not pass correctness")


if __name__ == "__main__":
    main()