from pydra import REQUIRED, Config
import argparse
import os

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestTimeScalingConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED # name of the run

        # Test-Time Scaling Method
        self.method = REQUIRED # "base", "best-of-N", "iterative refinement", "METR", "Cognition", "Stanford"
        self.num_parallel = 1 # used for best-of-N, METR, iterative refinement, stanford
        self.num_samples = 1 # used for METR
        self.num_iterations = 1 # used for iterative refinement, beam search
        self.num_best = 1 # used for beam search
        self.prompt = "regular" # "regular" or "cot"

        # Dataset
        self.dataset_src = "local" # either huggingface or local
        self.dataset_name = "ScalingIntelligence/KernelBench"

        # Problem Specification
        self.level = REQUIRED        
        # subset of problems to generate, otherwise generate on all problems in the level
        self.subset = (None, None) # (start_problem_id, end_problem_id), inclusive

        # Logging
        # Top Directory to Store Runs
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")
    
        self.verbose = False
        self.store_type = "local" # TODO: add Database Integration

        self.log_prompt = True
        self.log_response = True

        # GENERATION CONFIGS
        # num of thread pool to call inference server in parallel
        self.num_workers = 1
        self.api_query_interval = 0.0

        self.server_type = "openai"
        self.model_name = "gpt-4o-mini"
        self.max_tokens = 4096
        self.temperature = 1.0
        
        # EVALUATION CONFIGS
        self.eval_mode = "local"

        # Construct this from mapping from architecture name to torch cuda arch list in the future
        # you can either specify SM version or just use the name
        self.gpu_arch = ["Ada"]

        # Eval Flow setting
        # To speedup evaluation, you can start building the kernel on CPU on disk as cache
        self.build_cache_with_cpu = True
        self.num_cpu_workers = 1
        
        # Directory to build kernels for evaluation
        self.kernel_eval_build_dir = os.path.join(REPO_TOP_DIR, "cache")

        # number of GPUs to do batch evaluation
        self.num_gpu_devices = 1

        # Eval settings
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 180 # in seconds
        self.measure_performance = True

        self.hardware = "RTX_3090_Ti"
        
       

    def __repr__(self):
        return f"TestTimeScalingConfig({self.to_dict()})"
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--_tags", type=str, default="test")

    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, default=os.path.join(REPO_TOP_DIR, "runs"))

    # Methods
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="regular")
    parser.add_argument("--num_parallel", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--num_best", type=int, default=1)

    # Dataset
    parser.add_argument("--dataset_src", type=str, default="local")
    parser.add_argument("--dataset_name", type=str, default="ScalingIntelligence/KernelBench")
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--subset", type=str, default="(None, None)")

    # Inference Server
    parser.add_argument("--server_type", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--api_query_interval", type=float, default=0.0)

    # Eval
    parser.add_argument("--eval_mode", type=str, default="local")
    parser.add_argument("--gpu_arch", type=str, default="Ampere")
    parser.add_argument("--num_gpu_devices", type=int, default=1)

    parser.add_argument("--build_cache_with_cpu", type=bool, default=True)
    parser.add_argument("--num_cpu_workers", type=int, default=1)
    parser.add_argument("--kernel_eval_build_dir", type=str, default=os.path.join(REPO_TOP_DIR, "cache"))

    parser.add_argument("--num_correct_trials", type=int, default=5)
    parser.add_argument("--num_perf_trials", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--measure_performance", type=bool, default=True)

    # Logging
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--log_prompt", type=bool, default=True)
    parser.add_argument("--log_response", type=bool, default=True)
    parser.add_argument("--store_type", type=str, default="local")

    args = parser.parse_args()

    # Convert to same format as TestTimeScalingConfig.
    # TODO: this is hack for now, eventually migrate fully to argparse for wandb compatibility
    args.gpu_arch = args.gpu_arch.split(",")
    range_str = args.subset.strip("()").split(",")
    args.subset = (None, None) if range_str[0] == "None" else (int(range_str[0]), int(range_str[1]))
    return args