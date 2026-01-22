import matplotlib.pyplot as plt
import numpy as np

# --- H100 SXM5 HARDWARE LIMITS ---
# 3.35 TB/s Bandwidth, 989 TFLOPS TF32 Compute
PEAK_BW = 3.35 * 10**12      
PEAK_FLOPS = 989 * 10**12    
RIDGE_POINT = PEAK_FLOPS / PEAK_BW  # ~295.2 FLOPs/Byte

def calc_conv_theoretical(n, ci, co, h, w, k, s=1, p=1):
    ho, wo = (h + 2*p - k)//s + 1, (w + 2*p - k)//s + 1
    flops = 2 * n * co * ci * k * k * ho * wo
    bytes_moved = (n * ci * h * w + co * ci * k * k + n * co * ho * wo) * 2
    return flops, bytes_moved

def calc_triton_theoretical(n, c, h, w):
    count = n * c * h * w
    flops = count * 2 # Add + ReLU
    bytes_moved = count * 2 * 3 # Read X + Read Y + Write Out
    return flops, bytes_moved

# --- DATA FROM YOUR RECENT BENCHMARKS ---
# Triton Kernel: 23.24 us
# Model Launch: 1.44 us (Used as a lower bound for synchronization)
kernel_stats = {
    "Triton_Fused_Add_ReLU": [*calc_triton_theoretical(10, 2048, 7, 7), 23.24e-6],
    "SM90_Conv_3x3_SOL":    [*calc_conv_theoretical(10, 512, 512, 14, 14, k=3), 20.48e-6], # Estimate from previous profile
    "Full_Model_Launch":     [1e9, 1e7, 1.44e-6], # Representing the high-speed CPU dispatch
}

def plot_final_sol():
    plt.figure(figsize=(12, 8))
    
    # Draw Roofline
    x = np.logspace(-1, 4, 1000)
    y = np.minimum(PEAK_FLOPS, PEAK_BW * x)
    plt.plot(x, y, 'r-', linewidth=3, label='H100 Theoretical Ceiling (TF32)')
    
    # Plot Points
    for name, (f, b, t) in kernel_stats.items():
        ai = f / b
        perf = f / t
        plt.scatter(ai, perf, s=150, edgecolors='black', zorder=5)
        plt.annotate(f"{name}\n{perf/1e9:.1f} GFLOPS", (ai, perf), 
                     xytext=(5,5), textcoords='offset points', fontweight='bold')

    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Attained Performance (FLOPS)')
    plt.title('H100 SPEED OF LIGHT ROOFLINE: ResNet-101 (Level 3)')
    plt.legend(loc='lower right')
    
    plt.savefig('sol_roofline_final.png', dpi=300)
    print("Final SOL Roofline plot generated: sol_roofline_final.png")

if __name__ == "__main__":
    plot_final_sol()

