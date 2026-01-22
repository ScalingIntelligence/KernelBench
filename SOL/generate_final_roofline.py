import matplotlib.pyplot as plt
import numpy as np

# --- H100 SXM5 HARDWARE LIMITS (SXM5 80GB) ---
PEAK_BW = 3.35 * 10**12      # 3.35 TB/s (HBM3)
PEAK_FLOPS = 989 * 10**12    # 989 TFLOPS (TF32)
RIDGE_POINT = PEAK_FLOPS / PEAK_BW  # ~295.2 FLOPs/Byte

def calc_conv(n, ci, co, h, w, k, s=1, p=1):
    ho, wo = (h + 2*p - k)//s + 1, (w + 2*p - k)//s + 1
    flops = 2 * n * co * ci * k * k * ho * wo
    # Memory: Input + Weights + Output (using FP16/TF32 = 2 bytes)
    bytes_moved = (n * ci * h * w + co * ci * k * k + n * co * ho * wo) * 2
    return flops, bytes_moved

def calc_fused(n, c, h, w):
    count = n * c * h * w
    flops = count * 2 # Add + ReLU
    # Memory: Read Input + Read Identity + Write Output
    bytes_moved = count * 2 * 3 
    return flops, bytes_moved

# --- YOUR MEASURED DATA ---
# Third value is your 'CUDA time avg' converted to seconds
kernel_stats = {
    "Triton_Add_ReLU": [*calc_fused(10, 2048, 7, 7), 6.744e-6],
    "SM90_3x3_Conv":  [*calc_conv(10, 512, 512, 14, 14, k=3), 20.489e-6],
    "SM90_1x1_Conv":  [*calc_conv(10, 1024, 2048, 7, 7, k=1, p=0), 17.077e-6],
}

# --- PLOTTING LOGIC ---
plt.figure(figsize=(12, 8))
x = np.logspace(-1, 4, 1000)
y = np.minimum(PEAK_FLOPS, PEAK_BW * x)
plt.plot(x, y, 'r-', linewidth=2, label=f'H100 Theoretical Limit')

for name, (f, b, t) in kernel_stats.items():
    ai = f / b
    perf = f / t
    plt.scatter(ai, perf, s=100, zorder=5)
    plt.annotate(f"{name}\n{perf/1e12:.2f} TFLOPS", (ai, perf), xytext=(5,5), textcoords='offset points')

plt.xscale('log'); plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)'); plt.ylabel('Performance (FLOPS)')
plt.title('H100 FINAL ROOFLINE - ResNet-101 SOL')
plt.legend()
plt.savefig('roofline_plot.png')
print("Roofline plot updated with your exact profiler data!")

