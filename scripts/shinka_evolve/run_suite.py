import argparse
import subprocess
import sys
import time
import os
import datetime
from concurrent.futures import ThreadPoolExecutor

# From src/dataset.py - Level 1 Representative Subset
# These cover Matmul, Softmax, Norms, Convs, etc.
LEVEL_1_SUBSET = [
    23, # Softmax (High chance of speedup via fusion)
    26, # GELU (High chance of speedup)
    40, # LayerNorm (High chance of speedup)
    33, # BatchNorm
    1,  # Square Matmul (Hard)
    3,  # Batched Matmul
    6,  # Matmul Large K
    18, # Matmul Transposed
    36, # RMSNorm
    42, # Max Pool 2D
    48, # Mean Reduction
    54, # Conv 3D
    57, # Conv Transposed 2D
    82, # Conv Depthwise
    # Add more level 1 problems if desired, or loop 1-100
]

def run_problem(gpu_id, problem_id, args, results_root):
    print(f"[GPU {gpu_id}] Starting Level 1 Problem {problem_id}...")
    
    log_file = os.path.join(results_root, f"log_P{problem_id}.txt")
    
    cmd = [
        sys.executable, "scripts/shinka_evolve/run_search.py",
        "--level", "1",
        "--problem_id", str(problem_id),
        "--model", args.model,
        "--generations", str(args.generations),
        "--results_root", results_root,
        "--max_parallel_jobs", str(args.jobs_per_gpu)
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    start_time = time.time()
    
    with open(log_file, "w") as f:
        try:
            # We use check=True to raise exception on failure
            subprocess.run(
                cmd, 
                env=env, 
                check=True, 
                stdout=f, 
                stderr=subprocess.STDOUT
            )
            status = "✅ Success"
        except subprocess.CalledProcessError:
            status = "❌ Failed"
        except Exception as e:
            status = f"💥 Error: {e}"
            
    duration = time.time() - start_time
    print(f"[GPU {gpu_id}] Finished P{problem_id}: {status} ({duration:.1f}s)")
    return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-5-mini")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--gpus", type=str, default="5,6,7", help="Comma separated list of GPU IDs")
    parser.add_argument("--jobs_per_gpu", type=int, default=6, help="Parallel evals inside Shinka per GPU")
    args = parser.parse_args()

    # Create a timestamped root directory for this suite run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = f"runs/suite_lvl1_{args.model.replace('/', '_')}_{timestamp}"
    os.makedirs(results_root, exist_ok=True)

    gpu_list = [int(x) for x in args.gpus.split(",")]
    
    print(f"🚀 Starting ShinkaEvolve Suite")
    print(f"🤖 Model: {args.model}")
    print(f"🖥️  GPUs: {gpu_list}")
    print(f"📂 Results: {results_root}")
    print(f"🎯 Targets: {len(LEVEL_1_SUBSET)} problems")
    print("-" * 50)
    
    # Create a queue of problems
    # We use ThreadPoolExecutor. The worker threads just manage the subprocess calls.
    # The actual heavy lifting is done by the OS scheduling the python subprocesses onto the GPUs.
    
    with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
        # We need to map problems to GPUs as they become free.
        # This simple approach launches N futures where N = num_gpus.
        # Each future pulls from a shared iterator/queue.
        
        problem_queue = list(LEVEL_1_SUBSET)
        
        def gpu_worker(gpu_id):
            while problem_queue:
                # Simple thread-safe pop
                try:
                    pid = problem_queue.pop(0)
                except IndexError:
                    break
                run_problem(gpu_id, pid, args, results_root)

        # Launch one worker per GPU
        futures = [executor.submit(gpu_worker, gpu) for gpu in gpu_list]
        
        # Wait for all to finish
        for f in futures:
            f.result()

    print(f"\n🏁 Suite completed. Check {results_root} for results.")

if __name__ == "__main__":
    main()