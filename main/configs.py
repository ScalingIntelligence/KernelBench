from pydra import REQUIRED, Config
import os

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestTimeScalingConfig(Config):
    def __init__(self):
        self.run_name = REQUIRED # name of the run

        # Test-Time Scaling Method
        self.method = REQUIRED # "best-of-N", "iterative refinement", "METR", "Cognition", "Stanford"
        self.num_parallel = 1 # used for best-of-N, METR, iterative refinement, stanford
        self.num_samples = 1 # used for METR
        self.num_iterations = 1 # used for iterative refinement, beam search
        self.num_best = 1 # used for beam search

        # Dataset
        self.dataset_src = REQUIRED # either huggingface or local
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

        self.log_prompt = False 

        # GENERATION CONFIGS
        # num of thread pool to call inference server in parallel
        self.num_workers = 1
        self.api_query_interval = 0.0

        self.server_type = "deepseek"
        self.model_name = "deepseek-coder"
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
        
       

    def __repr__(self):
        return f"TestTimeScalingConfig({self.to_dict()})"
    
