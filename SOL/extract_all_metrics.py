import json
import glob
import os

print(f"{'Kernel ID':<10} | {'Name':<20} | {'Latency (us)':<15} | {'AI':<10}")
print("-" * 60)

# Paths to search
output_files = glob.glob("/home/sesterce/KernelBench-sol/outputs/*.json")

for file_path in sorted(output_files):
    # Skip the master results.json if it's not a specific kernel result
    if "results.json" in file_path: continue
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Extract kernel stats
            k_name = list(data['kernels'].keys())[0]
            k_data = data['kernels'][k_name]
            
            time_us = k_data['time_s'] * 1_000_000
            ai = k_data['ai']
            
            # Extract Level/Problem from filename (e.g., sol_metrics_level2_3.json)
            base_name = os.path.basename(file_path)
            parts = base_name.replace("sol_metrics_", "").replace(".json", "").split("_")
            k_id = f"L{parts[0][-1]} P{parts[1]}" if len(parts) >= 2 else "Unknown"

            print(f"{k_id:<10} | {k_name[:20]:<20} | {time_us:>12.3f} us | {ai:<10.3f}")
    except Exception as e:
        continue
