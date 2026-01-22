import torch
import matplotlib.pyplot as plt
import numpy as np

# --- H100 SXM5 HARDWARE LIMITS ---
# Reference: NVIDIA H100 Datasheet
PEAK_BW = 3.35 * 10**12      # 3.35 TB/s (HBM3)
PEAK_FLOPS = 989 * 10**12    # 989 TFLOPS (TF32)
RIDGE_POINT = PEAK_FLOPS / PEAK_BW # ~295.2 FLOPs/Byte

def calculate_conv_metrics(n, ci, co, h, w, k, s=1, p=1, precision=2):
    """Calculates theoretical FLOPs and Bytes for Conv2d."""
    ho, wo = (h + 2*p - k)//s + 1, (w + 2*p - k)//s + 1
    flops = 2 * n * co * ci * k * k * ho * wo
    # Memory Traffic: Input + Weights + Output
    bytes_moved = (n * ci * h * w + co * ci * k * k + n * co * ho * wo) * precision
    return flops, bytes_moved

def calculate_elementwise_metrics(n, c, h, w, ops_per_elem=2, precision=2):
    """Calculates metrics for fused Add+ReLU."""
    count = n * c * h * w
    flops = count * ops_per_elem
    # Memory Traffic: Read(x) + Read(identity) + Write(out)
    bytes_moved = count * precision * 3 
    return flops, bytes_moved

# --- DATA ENTRY FROM PROFILER ---
# Update the third value in each list with 'CUDA time avg' from get_exact_metrics.py
kernel_stats = {
    # Name: [FLOPs, Bytes, Measured_Time_Seconds]
    "Triton_Fused_Add_ReLU": [*calculate_elementwise_metrics(10, 2048, 7, 7), 6.8e-6],
    "SM90_Conv_Bottleneck_3x3": [*calculate_conv_metrics(10, 512, 512, 14, 14, k=3), 17.2e-6],
    "SM90_Conv_1x1_Projection": [*calculate_conv_metrics(10, 1024, 2048, 7, 7, k=1, p=0), 11.5e-6],
}

def plot_sol_roofline():
    plt.figure(figsize=(14, 9))
    
    # 1. Plot the Roofline Ceiling and Slope
    x_limit = np.logspace(-1, 4, 1000)
    y_limit = np.minimum(PEAK_FLOPS, PEAK_BW * x_limit)
    plt.plot(x_limit, y_limit, 'r-', linewidth=3, label=f'H100 Theoretical Limit (TF32)')
    
    # 2. Plot Kernel Points
    for name, (f, b, t) in kernel_stats.items():
        ai = f / b
        perf = f / t
        plt.scatter(ai, perf, s=150, edgecolors='black', zorder=5)
        plt.annotate(f"{name}\n({perf/1e12:.1f} TFLOPS)", 
                     (ai, perf), xytext=(8, 8), 
                     textcoords='offset points', fontsize=10, fontweight='bold')

    # 3. Formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.axvline(RIDGE_POINT, color='green', linestyle='--', alpha=0.5, label=f'Ridge Point ({RIDGE_POINT:.1f})')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)', fontsize=12)
    plt.ylabel('Attained Performance (FLOPS)', fontsize=12)
    plt.title('H100 Speed of Light Roofline: ResNet-101 Optimizations', fontsize=14)
    plt.legend(loc='lower right')
    
    plt.savefig('sol_roofline_final.png', dpi=300)
    print("Success: sol_roofline_final.png generated.")

if __name__ == "__main__":
    plot_sol_roofline()

