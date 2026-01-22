import json, os, glob
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Configuration ---
PEAK_BW = 3350.0  # GB/s for H100 PCIe (Adjust if using SXM)
PEAK_FLOPS = 989.0 # TFLOPS (FP16/BF16)
SOL_DIR = "/home/sesterce/KernelBench-sol/outputs"
# Path to your 100% completion baseline results
BASE_DIR = "/home/sesterce/KernelBench/results/baseline" 

def load_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
        # Handle KernelBench nested JSON structure
        k = list(data['kernels'].values())[0]
        return k['ai'], k['achieved_tflops'], data['module']

# --- 2. Data Collection ---
sol_files = glob.glob(f"{SOL_DIR}/sol_metrics_level*.json")
points = []

for sol_f in sol_files:
    # Identify problem (e.g., level2_3)
    name = os.path.basename(sol_f).replace("sol_metrics_", "").replace(".json", "")
    base_f = os.path.join(BASE_DIR, f"{name}_baseline.json")
    
    if os.path.exists(base_f):
        points.append({
            'name': name,
            'sol': load_data(sol_f),
            'base': load_data(base_f)
        })

# --- 3. Plotting ---
plt.figure(figsize=(12, 8))
x = np.logspace(-2, 2, 100)
plt.plot(x, np.minimum(PEAK_FLOPS, PEAK_BW * x / 1000), 'k-', alpha=0.8, label="H100 Roofline")

for p in points:
    # Plot Baseline
    plt.scatter(p['base'][0], p['base'][1], c='red', s=40, alpha=0.5)
    # Plot Sol
    plt.scatter(p['sol'][0], p['sol'][1], c='green', s=60, marker='*')
    # Draw connection line showing the shift
    plt.annotate('', xy=(p['sol'][0], p['sol'][1]), xytext=(p['base'][0], p['base'][1]),
                 arrowprops=dict(arrowstyle="->", color='gray', lw=1, alpha=0.3))

plt.xscale('log'); plt.yscale('log')
plt.xlabel('Arithmetic Intensity (FLOP/Byte)'); plt.ylabel('Performance (TFLOPS)')
plt.title('KernelBench: Sol vs Baseline (All Levels)')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.savefig(f"{SOL_DIR}/master_comparison_roofline.png")

