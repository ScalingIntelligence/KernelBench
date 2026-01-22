#!/usr/bin/env python3
"""
global_sol_runner.py

Batch-run SOL submissions under runs/sol_submission/<level>/ and record GPU latencies.

Usage:
  python3 global_sol_runner.py                 # runs default levels: level1,level2,level3
  python3 global_sol_runner.py --levels level1 level2 \
       --warmup 10 --iters 100 --out results.csv

Notes:
 - Requires torch and pandas installed in the active Python environment.
 - Modules are imported by file path (safe for filenames starting with digits).
 - The script is defensive: it tolerates missing Model/get_inputs and records failures in CSV.
"""
import argparse
import importlib.util
import os
import sys
import time
from typing import List, Any

try:
    import torch
except Exception as e:
    print("ERROR: torch not importable:", e, file=sys.stderr)
    raise

try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas not importable:", e, file=sys.stderr)
    raise

def safe_move_to_cuda(x: Any):
    if isinstance(x, torch.Tensor):
        return x.cuda()
    return x

def load_module_from_path(file_path: str):
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run_single_module(file_path: str, warmup: int, iters: int, model_iters: int, channels_last: bool):
    """Attempt to import, instantiate and time the Model() or call the triton kernel if present."""
    result = {
        "module_path": file_path,
        "module": os.path.splitext(os.path.basename(file_path))[0],
        "status": "UNKNOWN",
        "latency_us": None,
        "error": ""
    }

    try:
        mod = load_module_from_path(file_path)
    except Exception as e:
        result.update(status="FAIL_IMPORT", error=str(e))
        return result

    # Try to get model
    model = None
    if hasattr(mod, "Model"):
        try:
            model = mod.Model()
        except Exception as e:
            result.update(status="FAIL_INSTANTIATE", error=f"Model() error: {e}")
            return result
    else:
        result.update(status="NO_MODEL", error="Module has no Model()")
        return result

    # Move model to CUDA and eval
    try:
        model = model.cuda().eval()
        if channels_last:
            try:
                model = model.to(memory_format=torch.channels_last)
            except Exception:
                pass
    except Exception as e:
        result.update(status="FAIL_CUDA_MODEL", error=str(e))
        return result

    # Get inputs
    inputs = None
    if hasattr(mod, "get_inputs"):
        try:
            raw_inputs = mod.get_inputs()
            # normalize to list
            if isinstance(raw_inputs, (list, tuple)):
                inputs = [safe_move_to_cuda(t) for t in raw_inputs]
            elif raw_inputs is None:
                inputs = []
            else:
                inputs = [safe_move_to_cuda(raw_inputs)]
        except Exception as e:
            result.update(status="FAIL_GET_INPUTS", error=f"get_inputs() error: {e}")
            return result
    else:
        result.update(status="NO_INPUTS", error="Module has no get_inputs()")
        return result

    # Warmup / optional optimize() hook
    try:
        torch.cuda.synchronize()
        if hasattr(model, "optimize"):
            try:
                model.optimize(*inputs)
                torch.cuda.synchronize()
            except Exception:
                # Non-fatal; continue
                pass

        # Warmups
        for _ in range(warmup):
            _ = model(*inputs)
        torch.cuda.synchronize()
    except Exception as e:
        result.update(status="FAIL_DURING_WARMUP", error=str(e))
        return result

    # Timed runs
    try:
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_evt.record()
        for _ in range(iters):
            _ = model(*inputs)
        end_evt.record()
        torch.cuda.synchronize()

        avg_ms = start_evt.elapsed_time(end_evt) / float(iters)
        avg_us = avg_ms * 1000.0
        result.update(status="PASS", latency_us=avg_us)
    except Exception as e:
        result.update(status="FAIL_DURING_TIMING", error=str(e))
    finally:
        # Cleanup
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass

    return result

def run_batch_evaluation(base_dir: str, levels: List[str], warmup:int, iters:int, model_iters:int, channels_last: bool, out_csv: str):
    results = []
    print(f"\n{'Level':<10} | {'Kernel Name':<45} | {'Latency':<12} | {'Status'}")
    print("-" * 95)

    for lvl in levels:
        path = os.path.join(base_dir, lvl)
        if not os.path.exists(path):
            print(f"[WARN] Level directory not found: {path}")
            continue

        files = sorted([f for f in os.listdir(path) if f.endswith('.py') and not f.startswith('__')])
        for f in files:
            file_path = os.path.join(path, f)
            res = run_single_module(file_path, warmup=warmup, iters=iters, model_iters=model_iters, channels_last=channels_last)
            module_short = res["module"][:43]
            latency_str = f"{res['latency_us']:.2f} us" if res["latency_us"] else "ERROR"
            print(f"{lvl:<10} | {module_short:<45} | {latency_str:<12} | [{res['status']}]")
            results.append({
                "Level": lvl,
                "Module": res["module"],
                "ModulePath": res["module_path"],
                "Status": res["status"],
                "Latency_us": res["latency_us"] if res["latency_us"] else "",
                "Error": res["error"]
            })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"\nBatch Run Complete. Results saved to {out_csv}\n")
    return df

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", type=str, default="runs/sol_submission", help="base directory containing levelX folders")
    p.add_argument("--levels", nargs="+", default=["level1","level2","level3"], help="levels to scan")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument("--model-iters", type=int, default=100, help="(unused) kept for compatibility")
    p.add_argument("--channels-last", action="store_true", help="use channels_last memory format for inputs/model")
    p.add_argument("--out", type=str, default="batch_sol_results.csv", help="output CSV filename")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_batch_evaluation(args.base_dir, args.levels, args.warmup, args.iters, args.model_iters, args.channels_last, args.out)

