#!/usr/bin/env python3
import os
import csv
import time
import traceback
import importlib.util
import multiprocessing as mp
from typing import Any
import torch

# Configuration via Environment Variables
LEVEL = os.environ.get("LEVEL", "2")  # Change to 2, 3, or 4
BASE_PATH = os.environ.get("BASE_PATH", f"../../KernelBench/level{LEVEL}")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", f"h100_level{LEVEL}_efficiency_report.csv")
WARMUP_ITERS = int(os.environ.get("WARMUP_ITERS", "10"))
TIMED_ITERS = int(os.environ.get("TIMED_ITERS", "50"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_module_from_path(full_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _to_device_inputs(raw_inputs: Any, device: str):
    if isinstance(raw_inputs, dict):
        return True, {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in raw_inputs.items()}
    if torch.is_tensor(raw_inputs):
        return False, (raw_inputs.to(device),)
    if isinstance(raw_inputs, (list, tuple)):
        return False, tuple((v.to(device) if torch.is_tensor(v) else v) for v in raw_inputs)
    return False, (raw_inputs,)

def safe_model_init(module) -> Any:
    init_args = module.get_init_inputs() if hasattr(module, "get_init_inputs") else []
    if isinstance(init_args, dict):
        return module.Model(**init_args)
    return module.Model(*init_args) if isinstance(init_args, (list, tuple)) else module.Model()

def benchmark_worker(full_path, name, return_dict):
    """Isolated process to handle a single kernel/model."""
    try:
        from sol_model import get_h100_hardware_specs, calculate_sol
        from trace_workload import trace_problem_workload
        
        specs = get_h100_hardware_specs()
        mod = load_module_from_path(full_path, name)
        
        # 1. Trace Workload
        workload = trace_problem_workload(mod)
        f, b = float(workload.get("flops", 0.0)), float(workload.get("bytes", 0.0))
        
        # 2. Model Modeling
        prediction = calculate_sol(f, b, specs)
        
        # 3. Actual Measurement
        model = safe_model_init(mod).to(DEVICE).eval()
        
        # Enable H100 Performance Mode
        torch.set_float32_matmul_precision('high')
        try:
            # Attempt torch.compile for Level 2-4 fusions
            model = torch.compile(model)
        except:
            pass 

        raw_inputs = mod.get_inputs()
        is_dict, gpu_inputs = _to_device_inputs(raw_inputs, DEVICE)

        # Warmup
        with torch.no_grad():
            for _ in range(WARMUP_ITERS):
                model(**gpu_inputs) if is_dict else model(*gpu_inputs)
        torch.cuda.synchronize()

        # Timing
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        with torch.no_grad():
            for _ in range(TIMED_ITERS):
                model(**gpu_inputs) if is_dict else model(*gpu_inputs)
        end_event.record()
        torch.cuda.synchronize()
        
        avg_ms = start_event.elapsed_time(end_event) / float(TIMED_ITERS)
        
        # Pack Results
        return_dict['data'] = {
            "kernel": name, "sol_ms": prediction['sol_ms'], "actual_ms": avg_ms,
            "bound": prediction['bound'], "flops": f, "bytes": b
        }
    except Exception as e:
        return_dict['error'] = str(e)

def run_full_suite():
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), BASE_PATH))
    if not os.path.exists(abs_path):
        print(f"Error: Path {abs_path} not found.")
        return

    files = sorted([f for f in os.listdir(abs_path) if f.endswith(".py") and not f.startswith("__")])
    print(f"🚀 Universal H100 Benchmark | Level {LEVEL}")
    print(f"{'Kernel Name':<45} | {'SoL (ms)':<10} | {'Actual (ms)':<12} | {'Eff %':>6}")
    print("-" * 85)

    results = []
    for f in files:
        name = f.replace(".py", "")
        manager = mp.Manager()
        return_dict = manager.dict()
        
        p = mp.Process(target=benchmark_worker, args=(os.path.join(abs_path, f), name, return_dict))
        p.start()
        p.join(timeout=120) # 2 min timeout for heavy Level 4 models

        if 'data' in return_dict:
            d = return_dict['data']
            eff = (d['sol_ms'] / d['actual_ms']) * 100
            print(f"{name[:45]:<45} | {d['sol_ms']:10.4f} | {d['actual_ms']:12.4f} | {eff:6.1f}%")
            d['eff_percent'] = eff
            results.append(d)
        else:
            print(f"{name[:45]:<45} | FAILED or TIMEOUT")

        if p.is_alive():
            p.terminate()

    # Save to CSV
    if results:
        with open(OUTPUT_CSV, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    run_full_suite()

