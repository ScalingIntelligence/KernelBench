import torch

# --- CONFIGURATION: Update these for your new kernel ---
KERNEL_NAME = "Your_New_Kernel_Name"
MEASURED_TIME_MS = 0.035  # The median time from your benchmark
BASELINE_TIME_MS = 0.052  # The PyTorch baseline time

# --- METRICS CALCULATOR: Update based on your tensor size ---
# Example for a vector of size N (e.g., 2**20)
N = 1024 * 1024  
DTYPE_SIZE = 4 # 4 for float32, 2 for float16/bfloat16

# For element-wise (Add, ReLU, Dropout): 1 Read + 1 Write
# For Softmax: ~2 Reads + 1 Write (depending on fusion)
BYTES_MOVED = N * DTYPE_SIZE * 2 
FLOPS = N * 5 # Estimate based on ops (e.g. Softmax has exp, sum, div)

# --- H100 PCIe Constants ---
PEAK_BW = 2040e9        # 2.04 TB/s
PEAK_FLOPS = 51.22e12   # 51.22 TFLOPS (FP32)

def run_analysis():
    time_s = MEASURED_TIME_MS / 1000
    
    # Calculate Theoretical Minimum Time (SOL)
    t_mem = BYTES_MOVED / PEAK_BW
    t_comp = FLOPS / PEAK_FLOPS
    t_sol_s = max(t_mem, t_comp)
    
    # Efficiency and Speedup
    efficiency = (t_sol_s / time_s) * 100
    speedup = BASELINE_TIME_MS / MEASURED_TIME_MS
    achieved_tflops = (FLOPS / time_s) / 1e12
    achieved_gb_s = (BYTES_MOVED / time_s) / 1e9

    print(f"--- Analysis: {KERNEL_NAME} ---")
    print(f"Time:            {MEASURED_TIME_MS:.4f} ms")
    print(f"Speedup:         {speedup:.2f}x vs Baseline")
    print(f"Bytes Moved:     {BYTES_MOVED:,} bytes")
    print(f"Total FLOPs:     {FLOPS:,} ops")
    print("-" * 30)
    print(f"Achieved BW:     {achieved_gb_s:.2f} GB/s")
    print(f"Achieved TFLOPS: {achieved_tflops:.4f} TFLOPS")
    print(f"SOL Efficiency:  {efficiency:.2f}%")

if __name__ == "__main__":
    run_analysis()
