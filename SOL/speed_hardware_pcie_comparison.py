import json

# Your H100 PCIe Peak (for Efficiency calculation)
PCIE_BW = 2040e9 # 2.04 TB/s

def compare_to_baseline(optimized_json, baseline_json):
    with open(optimized_json, 'r') as f:
        opt = json.load(f)
    with open(baseline_json, 'r') as f:
        base = json.load(f)

    print(f"{'Kernel':<20} | {'Baseline (ms)':<15} | {'Triton (ms)':<15} | {'Speedup':<10} | {'% of SOL'}")
    print("-" * 80)

    for name, opt_data in opt['kernels'].items():
        # Get baseline time (assumes matching names in baseline file)
        base_time = base['kernels'].get(name, {}).get('time_s', 0)
        opt_time = opt_data['time_s']
        
        # Calculate Metrics
        speedup = base_time / opt_time if opt_time > 0 else 0
        
        # SOL Efficiency (Time it *should* take / Time it *did* take)
        t_sol = opt_data['bytes'] / PCIE_BW
        sol_pct = (t_sol / opt_time) * 100

        print(f"{name[:20]:<20} | {base_time*1000:<15.4f} | {opt_time*1000:<15.4f} | {speedup:<10.2f}x | {sol_pct:.1f}%")

# compare_to_baseline('sol_metrics_combined.json', 'baseline_results.json')

