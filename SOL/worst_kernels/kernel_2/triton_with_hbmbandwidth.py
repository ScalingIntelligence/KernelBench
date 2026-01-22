
def run_profile_and_save_trace(dataset, problem_id, num_trials=10):
    # ... [previous setup code for model/inputs] ...

    # 1. Enable Kineto for HBM Bandwidth tracking
    experimental_config = torch.profiler._utils._ExperimentalConfig(
        profiler_metrics=[
            "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        ],
        verbose=True
    )

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,  # Tracks memory allocation
        with_flops=True,      # Tracks TFLOPS
        experimental_config=experimental_config
    ) as prof:
        with record_function("h100_memory_analysis"):
            for _ in range(num_trials):
                model(*inputs)
                torch.cuda.synchronize()
                prof.step()

    # Print the memory-specific table
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
