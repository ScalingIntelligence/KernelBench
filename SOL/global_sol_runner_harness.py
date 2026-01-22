import importlib.util
import os
import pandas as pd
import sys

def run_batch_evaluation():
    # Target our submission directory
    base_dir = "runs/sol_submission"
    levels = ['level1', 'level2', 'level3']
    results = []

    print(f"\n{'Level':<10} | {'Kernel Name':<45} | {'Latency':<12} | {'Status'}")
    print("-" * 85)

    for lvl in levels:
        path = os.path.join(base_dir, lvl)
        if not os.path.exists(path):
            continue

        # Get all .py files, skipping __init__ and pycache
        files = sorted([f for f in os.listdir(path) if f.endswith('.py') and not f.startswith('__')])

        for f in files:
            module_name = f[:-3]
            file_path = os.path.join(path, f)
            
            try:
                # 1. Dynamic Import
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)

                # 2. Initialize Model and Inputs
                model = mod.Model().cuda().eval()
                inputs = mod.get_inputs()
                
                # Ensure all inputs are on GPU
                inputs = [i.cuda() if isinstance(i, torch.Tensor) else i for i in inputs]

                # 3. LEVEL 4: Optimization/Graph Capture
                # We pass the unpacked inputs to the optimize function
                torch.cuda.synchronize()
                if hasattr(model, 'optimize'):
                    model.optimize(*inputs)
                torch.cuda.synchronize()

                # 4. Measure Performance
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                # Warmup one more time
                for _ in range(10):
                    _ = model(*inputs)
                
                torch.cuda.synchronize()
                start_event.record()
                for _ in range(100):
                    # Unpack list to handle multiple arguments (A, B)
                    _ = model(*inputs) 
                end_event.record()
                
                torch.cuda.synchronize()
                avg_ms = start_event.elapsed_time(end_event) / 100
                avg_us = avg_ms * 1000
                
                print(f"{lvl:<10} | {module_name[:43]:<45} | {avg_us:>8.2f} us | [PASS]")
                results.append({"Level": lvl, "Kernel": module_name, "Latency_us": avg_us, "Status": "PASS"})

            except Exception as e:
                print(f"{lvl:<10} | {module_name[:43]:<45} | {'ERROR':>11} | [FAIL]")
                results.append({"Level": lvl, "Kernel": module_name, "Latency_us": 0, "Status": f"FAIL: {str(e)[:30]}"})
            
            # Clean up to prevent OOM during batch runs
            torch.cuda.empty_cache()

    # 5. Save Summary
    df = pd.DataFrame(results)
    df.to_csv("batch_sol_results.csv", index=False)
    print(f"\n{'='*30}")
    print(f"Batch Run Complete. Results saved to batch_sol_results.csv")
    print(f"{'='*30}\n")
    return df

if __name__ == "__main__":
    run_batch_evaluation()
