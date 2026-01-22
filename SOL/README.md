# SOL Data Dictionary (Ground Truth)

The following columns are contained in the analysis CSVs. These metrics are calibrated against the **H100 PCIe** baseline (2.04 TB/s BW / 51.22 TFLOPS).

| Column | Description |
| :--- | :--- |
| **kernel** | The name/ID of the GPU kernel. |
| **sol_ms** | **Theoretical Min Time:** Fastest possible time based on hardware limits. |
| **actual_ms** | **Measured Time:** Actual execution time on the H100. |
| **eff_percent** | **SOL Efficiency:** How close the kernel is to hardware limits ($sol\_ms / actual\_ms$). |
| **bound** | Indicates if the bottleneck is **Memory** or **Compute**. |
| **bytes** | Total data movement (Read + Write). |
| **flops** | Total floating-point operations executed. |
| **achieved_bw_gb_s** | Observed Memory Bandwidth in GB/s. |
| **achieved_flops_tflops** | Observed Compute Throughput in TFLOPS. |
| **bw_percent_of_peak** | Utilization of the 2.04 TB/s PCIe limit. |
| **flops_percent_of_peak** | Utilization of the 51.22 TFLOPS limit. |

