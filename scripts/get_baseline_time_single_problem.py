import torch
import numpy as np
from kernelbench.eval import (
    load_original_model_and_inputs,
    set_seed,
    fetch_ref_arch_from_problem_id,
)

from src.timing import get_timing_function, get_timing_stats

def measure_program_time(
        ref_arch_name: str,
        ref_arch_src: str, 
        num_trials: int = 100,
        timing_method: str="cuda_event",        
        use_torch_compile: bool = False,
        torch_compile_backend: str="inductor", 
        torch_compile_options: str="default",
        device: torch.device="cuda:0",
        verbose: bool = False,
) -> dict:
    """
    Measure the time of a KernelBench reference architecture
    """
    context = {}
    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(
        ref_arch_src, context
    )
    try:
        with torch.no_grad():
            torch.cuda.synchronize(device=device)
            set_seed(42)
            inputs = get_inputs()
            set_seed(42)
            init_inputs = get_init_inputs()
            inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in inputs
            ]
            init_inputs = [
                x.cuda(device=device) if isinstance(x, torch.Tensor) else x
                for x in init_inputs
            ]

            # Initialize PyTorch model, use this for eager mode execution
            model = Model(*init_inputs)
            
            if use_torch_compile:
                print(f"Using torch.compile to compile model {ref_arch_name} with {torch_compile_backend} backend and {torch_compile_options} mode")
                model = torch.compile(model, backend=torch_compile_backend, mode=torch_compile_options)
            else:
                print(f"Using PyTorch Eager Execution on {ref_arch_name}")
            
            model = model.cuda(device=device)
            torch.cuda.synchronize(device=device)
            timing_func = get_timing_function(timing_method )
            elapsed_times = timing_func(
                model, inputs, num_warmup=3, num_trials=num_trials, discard_first=1, verbose=verbose, device=device
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device)

            if verbose:
                print(f"{ref_arch_name} {runtime_stats}")
            
            return runtime_stats
    except Exception as e:
        print(f"[Eval] Error in Measuring Performance: {e}")

if __name__ == "__main__":
    ref_arch_name = "softmax"
    ref_arch_src = """
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=1)

batch_size = 4096
dim = 65536

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
    """
    print(measure_program_time(ref_arch_name, ref_arch_src, use_torch_compile=False, timing_method="cuda_event"))