import os
import json


# Your actual Sesterce H100 PCIe constants
PCIE_BW = 2040000000000.0   # 2.04 TB/s
PCIE_FLOPS = 51220000000000.0 # 51.22 TFLOPS

files = [
    "results.json", 
    "sol_metrics_combined.json", 
    "sol_metrics_from_model.json", 
    "sol_metrics_level2_3_softmax_dropout.json"
]

def calibrate_to_pcie(filename):
    if not os.path.exists(filename):
        return
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # 1. Update the theoretical maximums
    data['peak_bw'] = PCIE_BW
    data['peak_flops'] = PCIE_FLOPS
    
    # 2. Recalculate SOL Efficiency for each kernel
    if 'kernels' in data:
        for k_name, k_data in data['kernels'].items():
            # Theoretical time = max(data_movement / BW, computation / FLOPS)
            t_sol = max(k_data['bytes'] / PCIE_BW, k_data['flops'] / PCIE_FLOPS)
            
            # Efficiency = Theoretical Min Time / Actual Time
            k_data['sol_efficiency'] = (t_sol / k_data['time_s']) * 100
            
            # Update Achieved TFLOPS based on PCIe baseline
            k_data['pcie_utilization'] = (k_data['achieved_tflops'] / (PCIE_FLOPS / 1e12)) * 100
            
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Updated {filename} to H100 PCIe baseline.")

for f in files:
    calibrate_to_pcie(f)

