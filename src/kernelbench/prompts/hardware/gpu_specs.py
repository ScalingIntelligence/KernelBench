"""
A List of GPU Specs to include in the prompt

Supports both NVIDIA and AMD GPUs.
"""


# =============================================================================
# NVIDIA GPU Specifications
# =============================================================================

GPU_SPEC_INFO = {
    "L40S": {
        "GPU Architecture": "Ada",
        "GPU Memory": "48GB GDDR6 with ECC",
        "Memory Bandwidth": "864 GB/s",
        "RT Core Performance TFLOPS": "212",
        "FP32 TFLOPS": "91.6",
        "TF32 Tensor Core TFLOPS": "183.2 (366 with sparsity)",
        "FP16 Tensor Core TFLOPS": "362.05 (733 with sparsity)",
        "FP8 Tensor Core TFLOPS": "733 (1466 with sparsity)",
        "Peak INT8 Tensor TOPS": "733 (1466 with sparsity)",
        "Peak INT4 Tensor TOPS": "733 (1466 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    },
    "H100": {
        "GPU Architecture": "Hopper",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "3.35 TB/s",
        "FP64 TFLOPS": "34",
        "FP64 Tensor Core TFLOPS": "67",
        "FP32 TFLOPS": "67",
        "TF32 Tensor Core TFLOPS": "989 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "1979 with sparsity",
        "FP16 Tensor Core TFLOPS": "1979 with sparsity",
        "FP8 Tensor Core TFLOPS": "3958 with sparsity",
        "INT8 Tensor Core TOPS": "3958 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "228 KB",
        "Maximum shared memory per thread block": "227 KB",
    },
    # this is 40GB (Standard)
    "A100": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "40GB",
        "Memory Bandwidth": "1555 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "A100-80GB": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "80GB",
        "Memory Bandwidth": "1935 GB/s",
        "FP64 TFLOPS": "9.7",
        "FP64 Tensor Core TFLOPS": "19.5",
        "FP32 TFLOPS": "19.5",
        "TF32 Tensor Core TFLOPS": "156 (312 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "312 (624 with sparsity)",
        "FP16 Tensor Core TFLOPS": "312 (624 with sparsity)",
        "INT8 Tensor Core TOPS": "624 (1248 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    },
    "L4": {
        "GPU Architecture": "Ada",
        "GPU Memory": "24GB",
        "Memory Bandwidth": "300 GB/s",
        "FP32 TFLOPS": "30.3",
        "TF32 Tensor Core TFLOPS": "120 with sparsity",
        "BFLOAT16 Tensore Core TFLOPS": "242 with sparsity",
        "FP8 Tensor Core TFLOPS": "485 with sparsity",
        "INT8 Tensor Core TOPS": "485 with sparsity",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "24",
        "Shared memory capacity per SM": "100 KB",
        "Maximum shared memory per thread block": "99 KB",
    }, 
    "T4": {
        "GPU Architecture": "Turing",
        "GPU Memory": "16 GB GDDR6",
        "Memory Bandwidth": "300 GB/s",
        "Single-Precision TFLOPS": "8.1",
        "Mixed-Precision (FP16/FP32) TFLOPS": "65",
        "INT8 TOPS": "130",
        "INT4 TOPS": "260",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "16",
        "Shared memory capacity per SM": "64 KB",
    },
    "A10G": {
        "GPU Architecture": "Ampere",
        "GPU Memory": "24GB GDDR6",
        "Memory Bandwidth": "600 GB/s",
        "FP32 TFLOPS": "31.2",
        "TF32 Tensor Core TFLOPS": "62.5 (125 with sparsity)",
        "BFLOAT16 Tensore Core TFLOPS": "125 (250 with sparsity)",
        "FP16 Tensor Core TFLOPS": "125 (250 with sparsity)",
        "INT8 Tensor Core TOPS": "250 (500 with sparsity)",
        "INT4 Tensor Core TOPS": "500 (1000 with sparsity)",
        "Register File Size": "64K 32-bit registers per SM",
        "Maximum number of registers per thread": "255",
        "Maximum number of thread blocks per SM": "32",
        "Shared memory capacity per SM": "164 KB",
        "Maximum shared memory per thread block": "163 KB",
    }
}

# =============================================================================
# AMD GPU Specifications
# =============================================================================

AMD_GPU_SPEC_INFO = {
    # Based on provided rocminfo for AMD Radeon 9700 (gfx1201)
    "R9700": {
        "GPU Name": "AMD Radeon 9700 (gfx1201)",
        "GPU Architecture": "AMD RDNA4 (gfx1201)",
        "Compute Units": 64,
        "SIMDs per CU": 2,
        "Shader Engines": 4,
        "Shader Arrays per Engine": 2,
        "Wavefront Size": "Wave32",
        "Max Clock (MHz)": 2350,
        "Workgroup Max Size": 1024,
        "Max Waves per CU": 32,
        "Stream Processors": 4096,
        "Ray Accelerators": 64,
        "AI Accelerators": 128,
        "ROPs": 128,
        "Transistors": "53.9 Billion",
        "Peak Pixel Fill Rate": "373.76 GP/s",
        "L1 Cache": "32 KB",
        "L2 Cache": "8 MB",
        "L3 Cache": "64 MB",
        "Cacheline Size": "256 B",
        "LDS (Workgroup Local Memory)": "64 KB",
        "VRAM": "32,061,259,776 B (~29.85 GiB)",
        "Memory Bandwidth": "Unknown",
        "FP32 Vector TFLOPS": "47.8",
        "FP16 Vector TFLOPS": "95.7",
        "FP16 Matrix TFLOPS": "191 (383 w/ sparsity)",
        "FP8 Matrix TFLOPS": "383 (766 w/ sparsity)",
        "INT8 Matrix TOPS": "383 (766 w/ sparsity)",
        "INT4 Matrix TOPS": "766 (1531 w/ sparsity)",
        "Max Registers per Block": 196608,
        "Max Shared Memory per Block": 65536,
        "Max Threads per Block": 1024,
        "Max Threads per CU": 2048,
        "Shared Memory per CU": 2097152,
        "Warp Size": 32,
        "MFMA": "Unknown",
    },
    # Based on provided rocminfo for AMD Instinct MI355X (gfx950)
    "MI355X": {
        "GPU Name": "AMD Instinct MI355X (gfx950)",
        "GPU Architecture": "gfx950 (CDNA family)",
        "Compute Units": 256,
        "SIMDs per CU": 4,
        "Shader Engines": 32,
        "Shader Arrays per Engine": 1,
        "Wavefront Size": "Wave64",
        "Max Clock (MHz)": 2400,
        "Peak Engine Clock": "2.4 GHz",
        "Workgroup Max Size": 1024,
        "Max Waves per CU": 32,
        "Max Work-item per CU": 2048,
        "Matrix Cores": 1024,
        "Stream Processors": 16384,
        "L1 Cache": "32 KB",
        "L2 Cache": "4 MB",
        "L3 Cache": "256 MB",
        "Cacheline Size": "128 B",
        "LDS (Workgroup Local Memory)": "160 KB",
        "VRAM": "288 GB HBM3E (309,220,868,096 B)",
        "Memory Bandwidth": "8 TB/s",
        "Memory Interface": "8192 bits",
        "Infinity Cache (Last Level)": "256 MB",
        "FP16 Vector TFLOPS": "157.3",
        "FP16 Matrix PFLOPS": "2.5166 (5.0332 w/ sparsity)",
        "BF16 Matrix PFLOPS": "2.5166 (5.0332 w/ sparsity)",
        "INT8 Matrix POPS": "5.0332 (10.0664 w/ sparsity)",
        "MXFP8 PFLOPS": "5.0332",
        "OCP-FP8 PFLOPS": "5.0332 (10.0664 w/ sparsity)",
        "MXFP6 PFLOPS": "10.0663",
        "MXFP4 PFLOPS": "10.0663",
        "FP64 Vector TFLOPS": "78.6",
        "FP32 Vector TFLOPS": "157.3",
        "FP64 Matrix TFLOPS": "78.6",
        "FP32 Matrix TFLOPS": "157.3",
        "Max Registers per Block": 131072,
        "Max Shared Memory per Block": 163840,
        "Max Threads per Block": 1024,
        "Max Threads per CU": 2048,
        "Shared Memory per CU": 41943040,
        "Warp Size": 64,
        "MFMA": "Unknown",
    },
    # Based on provided rocminfo + HIP query for AMD Radeon PRO W7900D (gfx1100)
    "W7900D": {
        "GPU Name": "AMD Radeon PRO W7900D (gfx1100)",
        "GPU Architecture": "AMD RDNA3 (gfx1100)",
        "Compute Units": 96,
        "SIMDs per CU": 2,
        "Shader Engines": 6,
        "Shader Arrays per Engine": 2,
        "Wavefront Size": "Wave32",
        "Max Clock (MHz)": 1760,
        "Workgroup Max Size": 1024,
        "Max Waves per CU": 32,
        "Max Work-item per CU": 1024,
        "L1 Cache": "32 KB",
        "L2 Cache": "6 MB",
        "L3 Cache": "96 MB",
        "Cacheline Size": "128 B",
        "LDS (Workgroup Local Memory)": "64 KB",
        "VRAM": "Unknown",
        "Memory Bandwidth": "Unknown",
        "Max Registers per Block": 196608,
        "Max Shared Memory per Block": 65536,
        "Max Threads per Block": 1024,
        "Max Threads per CU": 2048,
        "Shared Memory per CU": 3145728,
        "Warp Size": 32,
        "MFMA": "Unknown",
    },
}

# =============================================================================
# GPU Concept Definitions
# =============================================================================

# Basic GPU concept definitions (NVIDIA-centric)
GPU_DEFINITIONS = {
    "Thread": "A thread is a single execution unit that can run a single instruction at a time.",
    "Thread Block": "A thread block is a group of threads that can cooperate with each other.",
    "Warp": "A warp is a group of threads that are scheduled together and execute in parallel.",
    "SM": "A Streaming Multiprocessor, the core execution unit on NVIDIA GPUs.",
    "Tensor Core": "Specialized units for mixed-precision matrix operations.",
    "Occupancy": "The ratio of active warps to the maximum supported on an SM.",
    "Shared Memory": "Shared memory is a memory space that can be accessed by all threads in a thread block.",
    "Shared Memory Bank": "A subdivision of shared memory that can cause bank conflicts.",
    "Register": "A register is a small memory space that can be accessed by a single thread.",
    "Global Memory": "Off-chip DRAM accessible by all threads on the GPU.",
    "Constant Memory": "Read-only cached memory optimized for uniform access.",
    "Coalesced Access": "Memory access pattern that combines multiple requests into fewer transactions.",
    "Divergence": "When threads in the same warp take different control paths.",
    "Memory Hierarchy": "Memory hierarchy is a pyramid of memory types with different speeds and sizes.",
    "Memory Bandwidth": "Memory bandwidth is the rate at which data can be read from or stored into memory.",
    "Cache": "Cache is a small memory space that stores frequently accessed data.",
    "HBM": "HBM is a high-bandwidth memory technology that uses 3D-stacked DRAM.",
}

# AMD GPU concept definitions
AMD_GPU_DEFINITIONS = {
    "Wavefront": "AMD's SIMD execution group (Wave32 or Wave64).",
    "Wave32": "A 32-lane wavefront, common on RDNA architectures.",
    "Wave64": "A 64-lane wavefront, common on CDNA architectures.",
    "Compute Unit (CU)": "AMD's equivalent of an NVIDIA SM.",
    "Work-item": "A single thread in a kernel execution.",
    "Workgroup": "A group of work-items that can synchronize and share LDS.",
    "SIMD": "A SIMD unit inside a CU that executes a wavefront.",
    "LDS": "Local Data Share, AMD's shared memory.",
    "VGPR": "Vector registers allocated per work-item.",
    "SGPR": "Scalar registers shared across a wavefront.",
    "Occupancy": "Number of active waves per CU, limited by registers and LDS.",
    "Infinity Cache": "AMD's last-level cache that reduces DRAM traffic.",
    "MFMA": "Matrix Fused Multiply-Add instruction for matrix cores.",
    "Barrier": "A workgroup synchronization point.",
}



# =============================================================================
# Best Practices
# =============================================================================

GPU_BEST_PRACTICES = [
    # From https://docs.nvidia.com/cuda/ada-tuning-guide/index.html
    # CUDA Best Practices Section
    "Find ways to parallelize sequential code.",
    "Minimize data transfers between the host and the device.",
    "Adjust kernel launch configuration to maximize device utilization.",
    "Ensure that global memory accesses are coalesced.",
    "Minimize redundant accesses to global memory whenever possible.",
    "Avoid long sequences of diverged execution by threads within the same warp.",
    "Use shared memory to cache data that is reused within a block.",
    "Avoid shared memory bank conflicts; pad arrays when needed.",
    "Balance occupancy against register and shared memory usage.",
    "Use vectorized loads/stores when they improve bandwidth.",
    "Prefer tensor cores for matrix operations when supported.",
    "Use streams to overlap compute and data transfers.",
    "Use asynchronous copy features (e.g., cp.async) when available.",
    # we added this to reference the specific GPU architecture
    "Use specialized instructions based on the specific GPU architecture",
]

AMD_GPU_BEST_PRACTICES = [
    "Prefer Wave32-friendly configurations on RDNA architectures.",
    "Prefer Wave64 on CDNA unless the kernel benefits from Wave32.",
    "Choose workgroup sizes as multiples of the wavefront (32 or 64).",
    "Start with workgroup sizes in [256, 512, 1024] for 1D kernels.",
    "Balance VGPR usage and occupancy; avoid register spilling.",
    "Use LDS for data reuse; pad to avoid LDS bank conflicts.",
    "Keep global memory access contiguous and aligned (128B where possible).",
    "Use vectorized loads/stores when it improves bandwidth utilization.",
    "Use MFMA/matrix cores for GEMM-like operations when available.",
    "Minimize divergent branches within a wavefront.",
    "Avoid fp16 for exp/log; cast to fp32 for numerically sensitive ops.",
]

# =============================================================================
# AMD-Specific Prompt Templates-In progress
# =============================================================================



# =============================================================================
# Helper Functions for GPU Detection
# =============================================================================

def get_gpu_vendor() -> str:
    """
    Detect the GPU vendor (nvidia or amd).
    Returns: 'nvidia', 'amd', or 'unknown'
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "unknown"
        # Check for HIP version (ROCm indicator)
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "amd"
        return "nvidia"
    except ImportError:
        return "unknown"


def get_gpu_specs_for_vendor(vendor: str) -> dict:
    """
    Get appropriate GPU specs dictionary based on vendor.
    """
    if vendor.lower() == "amd":
        return AMD_GPU_SPEC_INFO
    return GPU_SPEC_INFO


def get_gpu_definitions_for_vendor(vendor: str) -> dict:
    """
    Get appropriate GPU definitions dictionary based on vendor.
    """
    if vendor.lower() == "amd":
        return AMD_GPU_DEFINITIONS
    return GPU_DEFINITIONS


def get_gpu_best_practices_for_vendor(vendor: str) -> list:
    """
    Get appropriate best practices list based on vendor.
    """
    if vendor.lower() == "amd":
        return AMD_GPU_BEST_PRACTICES
    return GPU_BEST_PRACTICES