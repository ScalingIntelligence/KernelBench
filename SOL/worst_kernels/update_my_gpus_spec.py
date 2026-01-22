# gpuspecs.py
GPU_SPECS = {
    "NVIDIA H100 PCIe": {
        "architecture": "Hopper",
        "memory_bandwidth_gb_s": 2040,      # HBM2e Peak
        "fp32_tflops": 51.22,               # Standard CUDA Core Peak
        "tensor_tf32_tflops": 756,          # Tensor Core Peak (with sparsity)
        "max_shared_mem_per_block_kb": 227,
        "l2_cache_size_mb": 80
    }
}
