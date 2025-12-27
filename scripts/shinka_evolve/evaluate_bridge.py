import argparse
import sys
import os
import torch
import json
import traceback

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.utils import read_file, set_gpu_arch

def get_current_gpu_arch():
    """Dynamically detect GPU architecture to set correct NVCC flags."""
    if not torch.cuda.is_available():
        return ["Volta"] # Fallback
        
    cap = torch.cuda.get_device_capability(0)
    major, minor = cap
    
    if major == 9: return ["Hopper"]
    if major == 8 and minor == 9: return ["Ada"]
    if major == 8: return ["Ampere"]
    if major == 7 and minor == 5: return ["Turing"]
    if major == 7: return ["Volta"]
    
    return ["Ampere"] # Default fallback for modern cards

def main(program_path, results_dir, level, problem_id):
    os.makedirs(results_dir, exist_ok=True)
    
    if not os.path.exists(program_path):
        write_results(results_dir, False, "File not found", {})
        return

    device = torch.device("cuda:0")
    
    # FIX: Dynamic Architecture Detection
    # Prevents "no kernel image is available" errors on H100/A100
    current_arch = get_current_gpu_arch()
    set_gpu_arch(current_arch) 

    # Load Reference
    try:
        dataset = construct_kernelbench_dataset(level)
        ref_arch_src = read_file(dataset[problem_id - 1])
    except Exception as e:
        write_results(results_dir, False, f"Ref Load Error: {e}", {})
        return

    # Load Candidate
    with open(program_path, 'r') as f:
        custom_model_src = f.read()

    metrics = {"combined_score": 0.0, "text_feedback": ""}

    # FIX: Create a unique build directory for this specific evaluation run
    # This prevents race conditions on the global torch_extensions lock file
    # if parallel jobs are running.
    jit_build_dir = os.path.join(results_dir, "jit_build")
    os.makedirs(jit_build_dir, exist_ok=True)

    try:
        # Run Eval
        result = eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=custom_model_src,
            seed_num=42,
            num_correct_trials=5,
            num_perf_trials=100,
            measure_performance=True,
            timing_method="cuda_event",
            device=device,
            check_for_excessive_speedup=True,
            excessive_speedup_threshold=50.0,
            build_dir=jit_build_dir  # <--- Critical for concurrency
        )

        if not result.compiled:
            msg = result.metadata.get('compilation_error', 'Unknown Error')
            # Hint for the user if they messed up ordering
            if "name 'matmul' is not defined" in msg or "NameError" in msg:
                msg += "\n\nHINT: You must define your CUDA kernel variables BEFORE the class ModelNew uses them."
            
            metrics["text_feedback"] = f"Compilation/Runtime Error:\n{msg}"
            write_results(results_dir, False, "Compilation Failed", metrics)
        
        elif not result.correctness:
            metrics["text_feedback"] = f"Incorrect Output. Max Diff: {result.metadata.get('max_difference', 'N/A')}"
            write_results(results_dir, False, "Incorrect", metrics)

        else:
            runtime = max(result.runtime, 1e-9)
            ref_runtime = max(result.ref_runtime, 1e-9)
            speedup = ref_runtime / runtime
            
            metrics["combined_score"] = float(speedup)
            metrics["public"] = {"speedup": float(speedup), "runtime_ms": float(runtime)}
            metrics["text_feedback"] = f"Success! Speedup: {speedup:.2f}x"
            write_results(results_dir, True, None, metrics)

    except Exception as e:
        metrics["text_feedback"] = f"Harness Error:\n{str(e)}"
        write_results(results_dir, False, str(e), metrics)

def write_results(results_dir, correct, error_msg, metrics):
    with open(os.path.join(results_dir, "correct.json"), "w") as f:
        json.dump({"correct": correct, "error": error_msg}, f, indent=4)
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Eval Done. Correct: {correct}, Score: {metrics.get('combined_score', 0)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--problem_id", type=int, required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir, args.level, args.problem_id)