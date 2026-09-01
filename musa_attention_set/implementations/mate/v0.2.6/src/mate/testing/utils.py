import time
import math
import random
import numpy as np
import torch
import einops
import os
import sys
import functools

from mate.utils import ceil_div

from typing import Optional, Literal


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def per_token_cast_to_fp8(x: torch.Tensor, dtype):
    """
    Import from DeepGEMM
    """

    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    sf = x_amax / 448.0

    return (x_view * (1.0 / sf.unsqueeze(2))).to(dtype).view(m, n), sf


def per_block_cast_to_fp8(x: torch.Tensor, dtype):
    """
    Import from DeepGEMM
    """

    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (align(m, 128), align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = x_amax / 448.0
    x_scaled = (x_view * (1.0 / sf)).to(dtype)

    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def tensor_quantize_fp8(x: torch.Tensor, scale_type=None):
    if scale_type is None:
        scale_type = torch.float8_e4m3fn

    finfo = torch.finfo(scale_type)
    x_min, x_max = x.aminmax()

    amax = torch.max(x_min.abs(), x_max.abs()).clamp(min=1e-8)
    scale = finfo.max / amax

    x_scale_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)

    return x_scale_sat.to(scale_type), scale.float().reciprocal()


def group_quantize_fp8(x, scale_shape, tile_shape, out_dtype=None, scale_major=None):
    """
    Import from FlashInfer
    """

    if out_dtype is None:
        out_dtype = torch.float8_e4m3fn

    if scale_major is None:
        scale_major = "K"

    ndim = x.ndim
    assert ndim in [2, 3], f"x.ndim must be 2 or 3, but got {ndim}"
    assert ndim == len(scale_shape) == len(tile_shape)

    fp8_info = torch.finfo(out_dtype)
    fp8_amax = torch.tensor(fp8_info.max, device=x.device, dtype=torch.float32)

    # 2. Tiling and Scale Calculation
    if ndim == 2:
        s0, s1 = scale_shape
        t0, t1 = tile_shape
        if scale_major == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = einops.rearrange(
                x, "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1
            )
            abs_max = einops.reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(
                1e-4
            )
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = einops.repeat(
                x_scale, "s0 s1 -> (s0 t0) (s1 t1)", t0=t0, t1=t1
            )
        else:
            # Handle column-major tiling
            x_tiled = einops.rearrange(
                x, "(s1 t0) (s0 t1) -> s0 s1 t0 t1", s0=s0, s1=s1
            )
            abs_max = einops.reduce(x_tiled.abs(), "s0 s1 t0 t1 -> s0 s1", "max").clamp(
                1e-4
            )
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = einops.rearrange(x_scale, "s0 s1 -> s1 s0")
            scales_repeated = einops.repeat(
                scales_permuted, "s1 s0 -> (s1 t0) (s0 t1)", t0=t0, t1=t1
            )

    elif ndim == 3:
        s0, s1, s2 = scale_shape
        t0, t1, t2 = tile_shape
        if scale_major == "K":
            # Tile x and find the max absolute value in each tile
            x_tiled = einops.rearrange(
                x, "(s0 t0) (s1 t1) (s2 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = einops.reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Broadcast scales back to the original tensor shape
            scales_repeated = einops.repeat(
                x_scale, "s0 s1 s2 -> (s0 t0) (s1 t1) (s2 t2)", t0=t0, t1=t1, t2=t2
            )
        else:
            # Handle layout where the last two axes are swapped
            x_tiled = einops.rearrange(
                x, "(s0 t0) (s2 t1) (s1 t2) -> s0 s1 s2 t0 t1 t2", s0=s0, s1=s1, s2=s2
            )
            abs_max = einops.reduce(
                x_tiled.abs(), "s0 s1 s2 t0 t1 t2 -> s0 s1 s2", "max"
            ).clamp(1e-4)
            x_scale = abs_max / fp8_amax
            x_scale = torch.pow(2.0, torch.ceil(torch.log2(x_scale.abs())))

            # Permute scale axes before repeating to match layout
            scales_permuted = einops.rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1")
            scales_repeated = einops.repeat(
                scales_permuted,
                "s0 s2 s1 -> (s0 t0) (s2 t1) (s1 t2)",
                t0=t0,
                t1=t1,
                t2=t2,
            )

    # 3. Final Quantization
    # Divide the original tensor by the broadcasted scales
    x_fp32 = x / (scales_repeated + 1e-8)

    # Convert the result to the target FP8 format
    x_fp8 = x_fp32.to(out_dtype)

    return x_fp8, x_scale


def group_dequantize_fp8(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    scale_major: Optional[Literal["MN", "K"]] = None,
):
    """
    Import from FlashInfer
    """

    if x_scale.dim() == 0:
        return x.float() * x_scale

    if scale_major is None:
        scale_major = "K"

    ndim = x.ndim

    assert ndim in [2, 3], f"ndim of input tensor must be 2 or 3, got {ndim}"
    assert x.ndim == x_scale.ndim, (
        f"ndim of input tensor must be equal to ndim of scale tensor, got {x.ndim} and {x_scale.ndim}"
    )

    if ndim == 2:
        if scale_major == "K":
            # m k
            s0, s1 = x_scale.shape
        else:
            # k m
            s1, s0 = x_scale.shape

        x = einops.rearrange(
            x.to(torch.float32), "(s0 t0) (s1 t1) -> s0 s1 t0 t1", s0=s0, s1=s1
        )

        if scale_major == "K":
            x_scale = einops.rearrange(x_scale, "s0 s1 -> s0 s1 1 1")
        else:
            x_scale = einops.rearrange(x_scale, "s0 s1 -> s1 s0 1 1")

        out = einops.rearrange(x * x_scale, "s0 s1 t0 t1 -> (s0 t0) (s1 t1)")

    elif ndim == 3:
        if scale_major == "K":
            s0, s1, s2 = x_scale.shape
        else:
            s0, s2, s1 = x_scale.shape
        x = einops.rearrange(
            x.to(torch.float32),
            "(s0 t0) (s1 t1) (s2 t2)-> s0 s1 s2 t0 t1 t2",
            s0=s0,
            s1=s1,
            s2=s2,
        )
        if scale_major == "K":
            x_scale = einops.rearrange(x_scale, "s0 s1 s2 -> s0 s1 s2 1 1 1")
        else:
            x_scale = einops.rearrange(x_scale, "s0 s1 s2 -> s0 s2 s1 1 1 1")
        out = einops.rearrange(
            x * x_scale, "s0 s1 s2 t0 t1 t2 -> (s0 t0) (s1 t1) (s2 t2)"
        )

    return out


def make_deepgemm_contig_m_indices(
    nr_group, expect_m_per_group, mode="random", alignment=1, device="musa"
):
    """
    Create group assignment indices for DeepGEMM Group GEMM Contiguous.
    """

    if mode == "random":
        group_ms = [
            int(expect_m_per_group * random.uniform(0.7, 1.3)) for _ in range(nr_group)
        ]
        m = sum([ceil_div(x, alignment) * alignment for x in group_ms])
        m_indices = torch.empty((m,), dtype=torch.int32, device=device)

        start = 0
        for i, group_m in enumerate(group_ms):
            actual_end = start + group_m
            aligned_end = start + ceil_div(group_m, alignment) * alignment
            m_indices[start:actual_end] = i
            m_indices[actual_end:aligned_end] = -1
            start = aligned_end

    elif mode == "fixed":
        m = nr_group * expect_m_per_group
        m_indices = torch.arange(
            nr_group, device=device, dtype=torch.int32
        ).repeat_interleave(expect_m_per_group)
    else:
        raise ValueError(
            "make_deepgemm_m_indices get invalid mode! Only 'random' and 'fixed' are supported."
        )

    return m, m_indices


def make_deepgemm_masked_m(
    nr_group, expect_m_per_group, max_m, mode="random", device="musa"
):
    if mode == "random":
        m_lst = [
            min(max_m, int(expect_m_per_group * random.uniform(0.7, 1.3)))
            for _ in range(nr_group)
        ]
        valid_m = sum(m_lst)
        mask_m = torch.tensor(m_lst, device=device, dtype=torch.int32)

    elif mode == "fixed":
        assert expect_m_per_group <= max_m

        mask_m = torch.tensor(
            [expect_m_per_group for _ in range(nr_group)],
            device=device,
            dtype=torch.int32,
        )
        valid_m = nr_group * expect_m_per_group

    return valid_m, mask_m


def sleep_after_kernel_run(execution_time):
    """
    Sleep after kernel run. Dynamically adjust sleep time up to 1 sec based on execution time.

    Args:
        execution_time (float): Kernel execution time in milliseconds.

    Returns:
        None
    """

    """
    Import from FlashInfer
    """

    if not math.isinf(execution_time):
        sleep_time = np.min([execution_time / 200, 1.0])
    else:
        sleep_time = 0.01
    time.sleep(sleep_time)
    return


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str = None,
    flush_l2: bool = True,
    with_multiple_kernels: bool = False,
):
    # By default, flush L2 with an excessive 8GB memset to give the GPU some (literal) chill time without full idle
    flush_l2_size = int(8e9 // 4)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = suppress_stdout_stderr if suppress_kineto_output else empty_suppress
    with suppress():
        schedule = torch.profiler.schedule(
            wait=1, warmup=0, active=num_tests * 2 - 1, repeat=1
        )
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.MUSA], schedule=schedule
        )
        with profiler:
            for i in range(2):
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            flush_l2_size, dtype=torch.int, device="musa"
                        ).zero_()
                    fn()

                    profiler.step()

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tuple = isinstance(kernel_names, tuple)
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=100)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    if not with_multiple_kernels:
        for name in kernel_names:
            assert sum([name in line for line in prof_lines]) <= 1, (
                f"Errors of the kernel {name} in the profiling table"
            )

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        total_time: float = 0
        total_num: int = 0
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                num_str = line.split()[-1]
                for unit, scale in units.items():
                    if unit in time_str:
                        total_time += (
                            float(time_str.replace(unit, "")) / scale * int(num_str)
                        )
                        total_num += int(num_str)
                        break
        kernel_times.append(total_time / total_num if total_num > 0 else 0)

    return tuple(kernel_times) if is_tuple else kernel_times[0]


def bench_gpu_time_with_musa_event(
    fn,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: bool = True,
    l2_flush_size_mb: int = 512,
    l2_flush_device: str = "musa",
    sleep_after_run: bool = False,
):
    """
    Benchmark kernel execution time using MUSA events (no MUSA graphs).
    Measures kernel launch latency + actual kernel execution time for fn().
    Can flush L2 cache and sleep after the run.

    Number of dry run and actual run iterations can be set by iteration count or time:
    - If dry_run_iters and repeat_iters are provided, provided iteration count will be used.
    - If dry_run_iters and repeat_iters are not provided, dry_run_time_ms and repeat_time_ms will be used.

    Returns an array of measured times so that the caller can compute statistics.

    Args:
        fn: Function to benchmark.
        dry_run_iters: Number of dry runs during which times does not count. If not provided, dry_run_time_ms will be used.
        repeat_iters: Number of iterations. If not provided, repeat_time_ms will be used.
        dry_run_time_ms: Time to run the dry run in milliseconds.
        repeat_time_ms: Time to run the repeat in milliseconds.
        l2_flush: Whether to flush cache.
        l2_flush_size_mb: Size of the cache to flush.
        l2_flush_device: Device that needs to flush cache.
        sleep_after_run: Whether to sleep after the run. Sleep time is dynamically set.

    Returns:
        measured_times: List of measured times.
    """

    """
    Import from FlashInfer
    """

    start_event = torch.musa.Event(enable_timing=True)
    end_event = torch.musa.Event(enable_timing=True)
    if l2_flush:
        l2_flush_size = int(l2_flush_size_mb) * 1024 * 1024
        buffer = torch.zeros(l2_flush_size, device=l2_flush_device, dtype=torch.int8)
        buffer += 1

    ## Estimate kernel execution time by running the kernel 5 times
    measurement_iters = 5
    torch.musa.synchronize()
    fn()  # Call once to exclude initial overhead
    torch.musa.synchronize()
    start_event.record()
    for _ in range(measurement_iters):
        if l2_flush:
            buffer += 1
        fn()
    end_event.record()
    torch.musa.synchronize()
    estimated_kernel_execution_time = (
        start_event.elapsed_time(end_event) / measurement_iters
    )

    ## Set dry run and repeat iterations
    if dry_run_iters is None:
        dry_run_iters = max(1, int(dry_run_time_ms / estimated_kernel_execution_time))
    if repeat_iters is None:
        repeat_iters = max(1, int(repeat_time_ms / estimated_kernel_execution_time))

    # Dry runs
    if l2_flush:
        buffer.zero_()
    torch.musa.synchronize()
    for _ in range(dry_run_iters):
        if l2_flush:
            buffer += 1
        fn()
    torch.musa.synchronize()

    # Actual run
    if l2_flush:
        buffer.zero_()
    start_events = [torch.musa.Event(enable_timing=True) for _ in range(repeat_iters)]
    end_events = [torch.musa.Event(enable_timing=True) for _ in range(repeat_iters)]
    torch.musa.synchronize()
    for iter_idx in range(repeat_iters):
        if l2_flush:
            buffer += 1
        start_events[iter_idx].record()
        fn()
        end_events[iter_idx].record()

        if sleep_after_run:
            sleep_after_kernel_run(estimated_kernel_execution_time)

    # Synchronize once outside of the loop to avoid synchronization overhead
    torch.musa.synchronize()
    measured_times = []
    for iter_idx in range(repeat_iters):
        measured_times.append(start_events[iter_idx].elapsed_time(end_events[iter_idx]))
    return measured_times


def bench_gpu_time(
    fn,
    dry_run_iters: Optional[int] = None,
    repeat_iters: Optional[int] = None,
    dry_run_time_ms: int = 25,
    repeat_time_ms: int = 100,
    l2_flush: bool = True,
    l2_flush_size_mb: int = 512,
    l2_flush_device: str = "musa",
    sleep_after_run: bool = False,
    enable_mupti: bool = False,
    use_musa_graph: bool = False,
    num_iters_within_graph: int = 10,
):
    """
    Benchmark wrapper that chooses among MUPTI, MUSA events, or MUSA Graphs.

    By default, uses MUSA events (enable_mupti=False, use_musa_graph=False).

    Args mirror the underlying implementations; extra control flags:
    - enable_mupti: If True, use MUPTI to measure GPU kernel time.
    - If use_musa_graph is True, will capture and replay a MUSA graph during measurement.
    - use_musa_graph: If True (and enable_mupti is False), use MUSA graph timing.
    - num_iters_within_graph: Iterations to run within the MUSA graph when used (non-MUPTI path only).
    """

    return bench_gpu_time_with_musa_event(
        fn,
        dry_run_iters,
        repeat_iters,
        dry_run_time_ms,
        repeat_time_ms,
        l2_flush,
        l2_flush_size_mb,
        l2_flush_device,
        sleep_after_run,
    )


def check_gemm_sbo_signal(
    num_local_expert, max_m, block_m, threshold, signal, masked_m
):
    expert_len = ceil_div(max_m, block_m)

    for expert in range(num_local_expert):
        start = expert * expert_len
        end = expert * expert_len + expert_len

        mask = masked_m[expert]
        valid_len = ceil_div(mask, block_m)
        for i in range(start, end):
            if i < start + valid_len:
                assert signal[i] == threshold, (
                    f"check sbo signal failed! {i=}, {signal[i]=}, {threshold=}"
                )
            else:
                assert signal[i] == 0, f"check sbo signal failed! {i=}, {signal[i]=}"


def multidist_randn(
    num_dists, dim, mean_mean=0.0, mean_std=1.0, scale_lower=0.5, scale_upper=1.5
):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    scales = torch.distributions.Uniform(scale_lower, scale_upper).sample((num_dists,))
    data = torch.distributions.Normal(means, scales).sample((dim,))
    return data.T.contiguous()


def multidist_randu(num_dists, dim, mean_mean=0.0, mean_std=1.0, lower=-1.0, upper=1.0):
    means = torch.distributions.Normal(mean_mean, mean_std).sample((num_dists,))
    data = torch.distributions.Uniform(means + lower, means + upper).sample((dim,))
    return data.T.contiguous()


def gen_qkv(
    total_tokens,
    num_q_heads,
    num_k_heads,
    num_v_heads,
    dim_qk,
    dim_v,
    device,
    dtype=torch.float16,
):
    # qkv_rng = functools.partial(multidist_randn, mean_std=0.1)
    qkv_rng = functools.partial(multidist_randu, mean_std=0.05, lower=-0.25, upper=0.25)

    q = qkv_rng(total_tokens * num_q_heads, dim_qk)
    k = qkv_rng(total_tokens * num_k_heads, dim_qk)
    v = qkv_rng(total_tokens * num_v_heads, dim_v)

    q = q.reshape(total_tokens, num_q_heads, dim_qk).to(dtype).contiguous().to(device)
    k = k.reshape(total_tokens, num_k_heads, dim_qk).to(dtype).contiguous().to(device)
    v = v.reshape(total_tokens, num_v_heads, dim_v).to(dtype).contiguous().to(device)

    return q, k, v


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
