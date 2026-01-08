####
# Nsight Profiling Related Functions
# Currently showcases how to profile a kernelbench model with nsight-python
####

import os
import torch
import pandas as pd
import numpy as np

# wrapper with tool to measure hardware metric


# Check if nsight-python is available
# Reference: https://docs.nvidia.com/nsight-python
try:
    import nsight
    NSIGHT_AVAILABLE = True
    
except ImportError:
    NSIGHT_AVAILABLE = False


def profile_with_nsight(func, metrics=None, num_trials=1):
    """Profile a PyTorch function. Returns {metric_name: value}."""
    if not NSIGHT_AVAILABLE:
        raise RuntimeError("nsight-python not available")
    
    metrics = [metrics] if isinstance(metrics, str) else (metrics or ['sm__cycles_active.avg'])
    
    @nsight.analyze.kernel(
        metrics=metrics,
        runs=num_trials,
        configs=[(0,)],
        combine_kernel_metrics=lambda a, b: (a or 0) + (b or 0), # NOTE: some torch ops launch multiple kernels, so we need to combine metrics
    )
    def profiled(_):
        with nsight.annotate("kernel"):
            return func()
    
    try:
        result = profiled()
        if result is None:
            return {m: None for m in metrics}
        
        df = result.to_dataframe()
        if df is None or (hasattr(df, 'empty') and df.empty):
            return {m: None for m in metrics}
        
        if isinstance(df, pd.DataFrame) and 'Metric Name' in df.columns:
            return {row['Metric Name']: float(row['AvgValue']) for _, row in df.iterrows()}
        elif isinstance(df, pd.DataFrame):
            return {metrics[0]: float(df['AvgValue'].iloc[0])}
        else:
            # Handle case where df is not a DataFrame
            return {m: None for m in metrics}
    except Exception as e:
        print(f"Error profiling: {e}")
        import traceback
        traceback.print_exc()
        return {m: None for m in metrics}


# example function to profile with nsight
def example_ncu_python_profile():
    # Test with simple kernel
    def test_kernel(x, y):
        """Simple matmul kernel."""
        return x @ y
    
    print("Creating test tensors...")
    a = torch.randn(256, 256, device="cuda")
    b = torch.randn(256, 256, device="cuda")
    
    print("Running nsight profiling...")

    # Wrap kernel
    def test_kernel_forward():
        return test_kernel(a, b)

    metric_values = profile_with_nsight(
        test_kernel_forward,
        ['sm__cycles_active.avg', 'sm__cycles_elapsed.sum', "smsp__inst_executed_pipe_tensor_op_hmma.sum"],
        num_trials=1,
    )
    
    print("\nProfiling results:")
    for metric_name, value in metric_values.items():
        print(f"  {metric_name}: {value}")
    return
    

def check_ncu_available() -> bool:
    from shutil import which
    return which('ncu') is not None



# pytorch profiler
# migrate from old repo during ICML / caesar repo


def profile_kernelbench_model_with_nsight(
    custom_model_src: str,
    ref_model_src: str = None,
    metrics: list = None,
    num_trials: int = 1,
    seed: int = 42,
    device: torch.device = None,
    backend: str = "cuda",
    precision: torch.dtype = torch.float32,
    build_dir: str = None,
    verbose: bool = False,
) -> dict:
    """
    Profile a kernelbench model using nsight-python.
    
    Assumes model has been validated and compiled successfully via eval.
    No error checking is performed - this should only be called after eval proves correctness.
    
    Args:
        custom_model_src: Source code string for the custom model (ModelNew class)
        ref_model_src: Optional source code string for reference model to get get_inputs/get_init_inputs.
                      If None, will try to get these from custom_model_src.
        metrics: List of nsight metrics to collect. Default: ['sm__cycles_active.avg']
        num_trials: Number of trials to run. Default: 1
        seed: Random seed for reproducible inputs
        device: CUDA device to run on. Default: cuda:0
        backend: Backend type ('cuda', 'triton', 'tilelang', 'cute'). Default: 'cuda'
        precision: torch.dtype for computation. Default: torch.float32
        build_dir: Build directory for compiled kernels. Default: None
        verbose: Whether to print verbose output. Default: False
    
    Returns:
        Dictionary mapping metric names to their values
    """
    from kernelbench.eval import (
        load_custom_model,
        load_custom_model_with_tempfile,
        load_original_model_and_inputs,
        _process_input_tensor,
        set_seed,
        graceful_eval_cleanup,
    )
    from kernelbench.utils import set_gpu_arch
    
    device = device or torch.device("cuda:0")
    metrics = metrics or ['sm__cycles_active.avg']
    metrics = [metrics] if isinstance(metrics, str) else metrics
    
    # set_gpu_arch(["Ada"]) # NOTE: can set GPU arch here if needed
    
    torch.cuda.set_device(device)
    
    # Load input functions using existing eval function
    input_source = ref_model_src or custom_model_src
    context = {}
    _, get_init_inputs, get_inputs = load_original_model_and_inputs(input_source, context)
    
    # Prepare inputs
    set_seed(seed)
    init_inputs = [_process_input_tensor(x, device, backend, precision) for x in get_init_inputs()]
    
    # Load and compile model
    if verbose:
        print("[Profile] Loading and compiling custom model...")
    
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    tempfile = None
    
    if backend.lower() in ["triton", "tilelang", "cute"]:
        ModelNew, tempfile = load_custom_model_with_tempfile(custom_model_src, entry_point="ModelNew")
    else:
        ModelNew = load_custom_model(custom_model_src, {}, build_dir)
    
    torch.cuda.synchronize(device=device)
    
    # Instantiate model
    with torch.no_grad():
        set_seed(seed)
        custom_model = ModelNew(*init_inputs)
        custom_model = custom_model.to(device=device, dtype=precision)
        torch.cuda.synchronize(device=device)
    
    if verbose:
        print("[Profile] Model instantiated successfully")
    
    # Prepare profiling inputs
    set_seed(seed)
    inputs = [_process_input_tensor(x, device, backend, precision) for x in get_inputs()]
    
    # Profile
    if verbose:
        print(f"[Profile] Profiling with nsight (metrics: {metrics})...")
    
    # Wrap model forward pass
    def model_forward():
        with torch.no_grad():
            return custom_model(*inputs)
    
    # Profile with nsight
    metric_values = profile_with_nsight(model_forward, metrics=metrics, num_trials=num_trials)
    
    if verbose:
        print("[Profile] Profiling completed successfully")
    
    # Cleanup using existing eval function
    graceful_eval_cleanup(context, device, tempfile)
    
    return metric_values



if NSIGHT_AVAILABLE:
    @nsight.analyze.kernel
    def benchmark_matmul(n):
        """Standard benchmark following nsight-python docs."""
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        with nsight.annotate("matmul"):
            c = a @ b
        return c




###########
# 
############
def test_flash_attention_profile():
    """
    Test the profile_kernelbench_model_with_nsight function using the tiled_matmul example.
    """
    import os
    from kernelbench.utils import read_file
    
    # Get the paths to the reference and custom model files
    REPO_ROOT = os.path.dirname(__file__)
    ref_model_path = os.path.join(
        REPO_ROOT,
        "prompts/few_shot/model_ex_flash_attn.py"
    )
    custom_model_path = os.path.join(
        REPO_ROOT,
        "prompts/few_shot/model_new_ex_flash_attn.py"
    )
    
    # Read the model source files
    print("[Test] Reading model source files...")
    ref_model_src = read_file(ref_model_path)
    custom_model_src = read_file(custom_model_path)
    
    
    print("[Test] Starting profiling with nsight...")
    
    # Profile the custom model
    metrics = profile_kernelbench_model_with_nsight(
        custom_model_src=custom_model_src,
        ref_model_src=ref_model_src,
        metrics=[
            'gpu__time_duration.sum',
            'sm__cycles_elapsed.sum'
        ],
        seed=42,
        backend="cuda",
        precision=torch.float32,
        verbose=True
    )
    print(metrics)
    
    print("\n[Test] Profiling results:")
    print("=" * 60)
    for metric_name, value in metrics.items():
        if value is not None:
            print(f"  {metric_name}: {value}")
        else:
            print(f"  {metric_name}: <not available>")
    print("=" * 60)
    
    return metrics



if __name__ == "__main__":
    if not check_ncu_available():
        print("ncu not found in PATH. Install Nsight Compute.")
        exit(1)
    
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        exit(1)
        
    
    # test the example_ncu_python_profile, and flash attention profile
    example_ncu_python_profile()
    test_flash_attention_profile()
    