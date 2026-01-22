import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# --- 1. H100 PCIe Confirmed Specs ---
BW_HBM = 2040.0       # GB/s (HBM2e for PCIe version)
PEAK_FP16_TC = 756.0  # TFLOPS (Tensor Core)
PEAK_FP16_VEC = 151.0 # TFLOPS (Vector/CUDA Core)
PEAK_FP32 = 51.22     # TFLOPS (Standard FP32)

# --- 2. Data Aggregator ---
def gather_metrics(pattern):
    kernels_found = {}
    for f_path in glob.glob(pattern):
        try:
            with open(f_path, 'r') as f:
                data = json.load(f)
                # Handle standard KernelBench nested structure or flat structure
                kernels = data.get('kernels', {})
                for k_name, m in kernels.items():
                    # Calculate AI/TFLOPS if the file is missing them
                    if 'ai' not in m or 'achieved_tflops' not in m:
                        ai = m['flops'] / m['bytes']
                        tflops = (m['flops'] / m['time_s']) / 1e12
                    else:
                        ai, tflops = m['ai'], m['achieved_tflops']
                    
                    # Store unique entries
                    kernels_found[k_name] = {'ai': ai, 'tflops': tflops}
        except: continue
    return kernels_found

# Collect all data
baselines = gather_metrics('results.json')
# Add level-specific baselines if you have them (e.g. baseline_level1.json)
baselines.update(gather_metrics('baseline_level*.json')) 

sols = gather_metrics('sol_metrics_level*.json')
sols.update(gather_metrics('sol_metrics_combined.json'))

# --- 3. Unified Plotting ---
fig, ax = plt.subplots(figsize=(14, 10))
x_ai = np.logspace(-2, 2, 1000)

# Rooflines (Hardware Physical Limits)
ax.plot(x_ai, np.minimum(PEAK_FP16_TC, (BW_HBM * x_ai) / 1000), 'r-', lw=2, label="H100 PCIe FP16 TC")
ax.plot(x_ai, np.minimum(PEAK_FP16_VEC, (BW_HBM * x_ai) / 1000), 'g--', alpha=0.6, label="H100 PCIe Vector")
ax.plot(x_ai, np.minimum(PEAK_FP32, (BW_HBM * x_ai) / 1000), 'b:', alpha=0.5, label="H100 PCIe FP32")

# Plot Baseline Kernels
for name, p in baselines.items():
    ax.scatter(p['ai'], p['tflops'], color='red', s=80, marker='o', alpha=0.7, zorder=5)
    ax.annotate(name, (p['ai'], p['tflops']), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=7, color='red')

# Plot SOL Kernels
for name, p in sols.items():
    ax.scatter(p['ai'], p['tflops'], color='gold', edgecolor='black', s=180, marker='*', zorder=6)
    ax.annotate(name, (p['ai'], p['tflops']), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, fontweight='bold')

# Legend & Formatting
ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which="both", alpha=0.2)
ax.set_title(f'KernelBench Full Submission Roofline: {len(sols)} Kernels Optimized', fontsize=16)
ax.set_xlabel('Arithmetic Intensity (FLOP/Byte)'); ax.set_ylabel('Throughput (TFLOPS)')

# Fix redundant legend labels
handles, labels = ax.get_legend_handles_labels()
unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
ax.legend(*zip(*unique), loc='lower right')

plt.tight_layout()
plt.savefig("full_submission_roofline.png", dpi=300)
