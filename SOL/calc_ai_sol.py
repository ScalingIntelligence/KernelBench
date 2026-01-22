mport torch
import matplotlib.pyplot as plt
import numpy as np

# --- H100 SXM5 HARDWARE LIMITS ---
PEAK_BW = 3.35 * 10**12      # 3.35 TB/s (HBM3)
PEAK_FLOPS = 989 * 10**12    # 989 TFLOPS (TF32)
RIDGE_POINT = PEAK_FLOPS / PEAK_BW # ~295 FLOPs/Byte

def calculate_conv_metrics(n, ci, co, h, w, k, s=1, p=1, precision=2):
    """
    Calculates exact FLOPs and Bytes for a Conv2d layer.
    precision: 2 bytes for TF32/FP16, 4 for FP32.
    """
    ho, wo = (h + 2*p - k)//s + 1, (w + 2*p - k)//s + 1
    # Standard MAC FLOPs (Multiply-Accumulate)
    flops = 2 * n * co * ci * k * k * ho * wo
    # Bytes: Input + Weights + Output
    bytes_moved = (n * ci * h * w + co * ci * k * k + n * co * ho * wo) * precision
    return flops, bytes_moved

def calculate_elementwise_metrics(n, c, h, w, ops_per_elem=1, precision=2):
    """Calculates metrics for Add or ReLU."""
    count = n * c * h * w
    flops = count * ops_per_elem
    # Read + Write (or Read+Read+Write for Add)
    bytes_moved = count * precision * (3 if ops_per_elem > 1 else 2)
    return flops, bytes_moved

# --- DATA ENTRY ---
# Replace 'measured_time' with the 'CUDA time avg' from your get_metrics.py report
kernel_stats = {
    # Name: [FLOPs, Bytes, Measured_Time_Seconds]
    "Triton_Fused_Add_ReLU": [*calculate_elementwise_metrics(10, 2048, 7, 7, ops_per_elem=2), 6.8e-6],
    "SM90_Conv_Bottleneck_3x3": [*calculate_conv_metrics(10, 512, 512, 14, 14, k=3), 17.2e-6],
    "SM90_Conv_1x1_Projection": [*calculate_conv_metrics(10, 1024, 2048, 7, 7, k=1, p=0), 11.5e-6],
}

# --- PLOTTING ---
plt.figure(figsize=(12, 8))
ai_vals, perf_vals, labels = [], [], []

for name, (f, b, t) in kernel_stats.items():
    ai = f / b
    perf = f / t
    ai_vals.append(ai)
    perf_vals.append(perf)
    labels.append(name)

# Draw Roofline
x = np.logspace(-2, 4, 1000)
y = np.minimum(PEAK_FLOPS, PEAK_BW * x)
plt.plot(x, y, 'r-', label=f'H100 Theoretical Limit (Ridge: {RIDGE_POINT:.1f})')

# Plot Points
plt.scatter(ai_vals, perf_vals, color='blue', s=100, zorder=5)
for i, txt in enumerate(labels):
    plt.annotate(txt, (ai_vals[i], perf_vals[i]), xytext=(5,5), textcoords='offset points', fontsize=9)

plt.xscale('log')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
plt.ylabel('Performance (FLOPS)')
plt.title('H100 SPEED OF LIGHT ROOFLINE: ResNet-101 Level 3')
plt.legend()
plt.savefig('sol_roofline_final.png')
print("Final SOL Roofline Plot saved as sol_roofline_final.png")
