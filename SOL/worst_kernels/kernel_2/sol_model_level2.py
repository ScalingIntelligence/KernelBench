mport os
import sys
import torch
import math
from trace_workload import trace_problem_workload

# --- H100 PCIe Specifications (Verified) ---
# PCIe version usually hits ~1,513 TFLOPS on FP16 Tensor Cores
# Memory Bandwidth is 2,000 GB/s (2.0 TB/s)
H100_PEAK_FLOPS = 1513.0 * (10**12)  
H100_MEM_BW_BYTES = 2000.0 * (10**9) 

def calculate_sol(flops, bytes_val):
    f = float(flops)
    b = float(bytes_val)
    
    if f <= 0 and b <= 0:
        return {"sol_ms": 0.0, "bound": "Unknown", "ai": 0.0}

    # Roofline: Time = max(Workload/ComputePeak, Workload/BandwidthPeak)
    theoretical_comp_s = f / H100_PEAK_FLOPS
    theoretical_mem_s = b / H100_MEM_BW_BYTES
    
    sol_s = max(theoretical_comp_s, theoretical_mem_s)
    sol_ms = sol_s * 1000.0
    
    ai = f / b if b > 0 else float('inf')
    bound = "Compute" if theoretical_comp_s > theoretical_mem_s else "Memory"
    
    return {"sol_ms": sol_ms, "bound": bound, "ai": ai}

def get_level_2_problems():
    # Targeted Level 2 path discovered in your directory
    base_path = "/root/KernelBench/KernelBench/level2"
    if not os.path.exists(base_path):
        print(f"Error: Could not find level2 directory at {base_path}")
        return []

    class Problem:
        def __init__(self, name, path):
            self.name = name
            self.path = path

    files = sorted([f for f in os.listdir(base_path) if f.endswith(".py") and f != "__init__.py"])
    return [Problem(f, os.path.join(base_path, f)) for f in files]

def run_analysis():
    print(f"--- Running NVIDIA H100 Level 2 SoL Analysis ---")
    problems = get_level_2_problems()
    
    print(f"{'Kernel Name':<40} | {'AI':<10} | {'SoL (ms)':<12} | {'Bottleneck'}")
    print("-" * 80)
    
    for problem in problems:
        try:
            # We import the module to pass it to the tracer
            import importlib.util
            spec = importlib.util.spec_from_file_location("mod", problem.path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            workload = trace_problem_workload(module)
            res = calculate_sol(workload.get("flops", 0), workload.get("bytes", 0))
            
            print(f"{problem.name:<40} | {res['ai']:<10.2f} | {res['sol_ms']:<12.6f} | {res['bound']}")
        except Exception as e:
            continue

if __name__ == "__main__":
    run_analysis()

