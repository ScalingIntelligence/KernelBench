import json

# Correct H100 PCIe Constants for your Sesterce VM
PCIE_BW = 2040e9  # 2.04 TB/s
PCIE_FLOPS = 51.22e12 # 51.22 TFLOPS

def compare_to_pcie_baseline(triton_file, baseline_file):
    with open(triton_file, 'r') as f:
        triton = json.load(f)
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)

    print(f"{'Kernel Name':<20} | {'Baseline (ms)':<15} | {'Triton (ms)':<15} | {'Speedup':<10} | {'% SOL'}")
    print("-" * 85)

    for name, t_data in triton['kernels'].items():
        # Get matching baseline time
        b_time = baseline['kernels'].get(name, {}).get('time_s', 0)
        t_time = t_data['time_s']
        
        # Calculate Speedup and Efficiency
        speedup = b_time / t_time if t_time > 0 else 0
        
        # SOL Time = max(Bytes / BW, FLOPs / TFLOPS)
        t_sol = max(t_data['bytes'] / PCIE_BW, t_data['flops'] / PCIE_FLOPS)
        efficiency = (t_sol / t_time) * 100

        print(f"{name[:20]:<20} | {b_time*1000:<15.4f} | {t_time*1000:<15.4f} | {speedup:<10.2f}x | {efficiency:.1f}%")

# Usage: compare_to_pcie_baseline('sol_metrics_combined.json', 'results.json')

