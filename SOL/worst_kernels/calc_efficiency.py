# Constants for H100 SXM5
THEORETICAL_PEAK_BW = 3350  # GB/s (3.35 TB/s)

# Metrics from your specific profile: 10 batches, 2048 channels, 7x7 map
# Data type: FP16/TF32 (2 bytes)
n, c, h, w = 10, 2048, 7, 7
precision = 2 

# Fused Add + ReLU: Read(x) + Read(identity) + Write(out) = 3 movements
total_bytes = (n * c * h * w) * precision * 3
measured_time_s = 6.744e-6  # From your 'CUDA time avg'

achieved_bw = (total_bytes / measured_time_s) / 1e9
efficiency = (achieved_bw / THEORETICAL_PEAK_BW) * 100

print(f"--- H100 Triton Efficiency Report ---")
print(f"Total Data Moved: {total_bytes / 1e6:.2f} MB")
print(f"Measured Time:    {measured_time_s * 1e6:.3f} us")
print(f"Achieved BW:      {achieved_bw:.2f} GB/s")
print(f"SOL Efficiency:   {efficiency:.2f}%")

