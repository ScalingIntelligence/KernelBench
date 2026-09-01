import numpy as np
import torch

from mate.utils import ceil_div
from mate.testing.utils import (
    bench_gpu_time,
    group_quantize_fp8,
    make_deepgemm_contig_m_indices,
    make_deepgemm_masked_m,
)
from mate.deep_gemm import (
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
    k_grouped_fp8_gemm_tn_contiguous,
    k_grouped_bf16_gemm_tn_contiguous,
)


def bench_group_deepgemm_fp8_nt_groupwise(
    nr_group, m_per_group, n, k, select_tile_m, in_dtype, out_dtype, verbose=True
):
    m, m_indices = make_deepgemm_contig_m_indices(
        nr_group, m_per_group, "random", select_tile_m
    )

    a = torch.rand((m, k), device="musa", dtype=torch.float)
    b = torch.rand((nr_group, n, k), device="musa", dtype=torch.float)

    d = torch.empty((m, n), device="musa", dtype=out_dtype)

    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, k // quant_tile)
    scale_b_shape = (nr_group, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, in_dtype, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, in_dtype, "K"
    )

    measurements = bench_gpu_time(
        lambda: m_grouped_fp8_gemm_nt_contiguous(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            d,
            m_indices,
            scale_granularity_mnk,
            alignment_m=select_tile_m,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        l2_flush=True,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    memory_bandwidth_per_second = (
        sum(
            [
                _.numel() * _.element_size()
                for _ in [fp8_a, fp8_b, scale_a, scale_b, m_indices, d]
            ]
        )
        * 1e-6
        / ms
    )

    if verbose:
        print(
            f"group_deepgemm_fp8_nt_groupwise nr_group={nr_group} m={m} n={n} k={k} select_tile_m={select_tile_m}\n"
            f"in_dtype={in_dtype} out_dtype={out_dtype}: \n"
            f"===> {tflops_per_second:.4f} TFLOPs/s\n"
            f"===> {memory_bandwidth_per_second:.4f} GB/s"
        )
        print()

    return tflops_per_second, memory_bandwidth_per_second


def bench_batch_deepgemm_fp8_nt_groupwise(
    nr_group,
    max_m_per_group,
    n,
    k,
    expected_m_per_group,
    select_tile_m,
    in_dtype,
    out_dtype,
    verbose=True,
):
    valid_m, masked_m = make_deepgemm_masked_m(
        nr_group, expected_m_per_group, max_m_per_group, "fixed"
    )

    a = torch.rand((nr_group, max_m_per_group, k), device="musa", dtype=torch.float)
    b = torch.rand((nr_group, n, k), device="musa", dtype=torch.float)

    d = torch.empty((nr_group, max_m_per_group, n), device="musa", dtype=out_dtype)

    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)
    quant_tile_shape_a = (1, 1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (nr_group, max_m_per_group, k // quant_tile)
    scale_b_shape = (nr_group, n // quant_tile, k // quant_tile)
    fp8_a, scale_a = group_quantize_fp8(
        a, scale_a_shape, quant_tile_shape_a, in_dtype, "K"
    )
    fp8_b, scale_b = group_quantize_fp8(
        b, scale_b_shape, quant_tile_shape_b, in_dtype, "K"
    )

    measurements = bench_gpu_time(
        lambda: m_grouped_fp8_gemm_nt_masked(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            d,
            masked_m,
            expected_m_per_group,
            scale_granularity_mnk,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        l2_flush=True,
    )
    ms = np.median(measurements)

    tflops_per_second = 2 * valid_m * n * k * 1e-9 / ms
    total_bytes = valid_m * k + nr_group * n * k + valid_m * n * 2
    total_bytes += (
        valid_m * (k // quant_tile) + scale_b.numel() + masked_m.numel()
    ) * 4
    memory_bandwidth_per_second = total_bytes * 1e-6 / ms

    if verbose:
        print(
            f"batch_deepgemm_fp8_nt_groupwise nr_group={nr_group} max_m={max_m_per_group} expected_m = {expected_m_per_group} valid_m = {valid_m} n={n} k={k} select_tile_m={select_tile_m}\n"
            f"in_dtype={in_dtype} out_dtype={out_dtype}: \n"
            f"===> {tflops_per_second:.4f} TFLOPs/s\n"
            f"===> {memory_bandwidth_per_second:.4f} GB/s"
        )
        print()

    return tflops_per_second, memory_bandwidth_per_second


def bench_k_grouped_contig_fp8_tn_groupwise(
    ks_per_group,
    m,
    n,
    in_dtype,
    out_dtype,
    verbose=True,
):
    quant_tile = 128
    scale_granularity_mnk = (1, 1, quant_tile)

    k = sum(ks_per_group)
    num_expert = len(ks_per_group)
    a_ = []
    b_ = []
    scale_a_ = []
    scale_b_ = []
    d = torch.rand((num_expert, m, n), device="musa", dtype=out_dtype)
    for i in range(num_expert):
        if ks_per_group[i] == 0:
            continue
        a = torch.rand(
            (m, ceil_div(ks_per_group[i], quant_tile) * quant_tile),
            device="musa",
            dtype=torch.float,
        )
        b = torch.rand(
            (n, ceil_div(ks_per_group[i], quant_tile) * quant_tile),
            device="musa",
            dtype=torch.float,
        )
        quant_tile_shape_a = (1, quant_tile)
        quant_tile_shape_b = (1, quant_tile)
        scale_a_shape = (m, ceil_div(ks_per_group[i], quant_tile))
        scale_b_shape = (n, ceil_div(ks_per_group[i], quant_tile))
        fp8_a, scale_a = group_quantize_fp8(
            a, scale_a_shape, quant_tile_shape_a, in_dtype, "K"
        )
        fp8_b, scale_b = group_quantize_fp8(
            b, scale_b_shape, quant_tile_shape_b, in_dtype, "K"
        )
        a_.append(fp8_a[:, : ks_per_group[i]].transpose(1, 0).contiguous())
        b_.append(fp8_b[:, : ks_per_group[i]].transpose(1, 0).contiguous())
        scale_a_.append(scale_a.transpose(1, 0).contiguous())
        scale_b_.append(scale_b.transpose(1, 0).contiguous())

    group_k_idx = torch.tensor(ks_per_group, device="musa", dtype=torch.int32)

    fp8_a = torch.cat(a_, dim=0).contiguous()
    scale_a = torch.cat(scale_a_, dim=0).contiguous()
    fp8_b = torch.cat(b_, dim=0).contiguous()
    scale_b = torch.cat(scale_b_, dim=0).contiguous()

    measurements = bench_gpu_time(
        lambda: k_grouped_fp8_gemm_tn_contiguous(
            (fp8_a, scale_a),
            (fp8_b, scale_b),
            d,
            None,
            group_k_idx,
            scale_granularity_mnk,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        l2_flush=True,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    total_bytes = m * k + n * k + num_expert * m * n * 2 * 4
    total_bytes += (scale_a.numel() + scale_b.numel() + group_k_idx.numel()) * 4
    memory_bandwidth_per_second = total_bytes * 1e-6 / ms

    if verbose:
        print(
            f"bench_k_grouped_contig_fp8_tn_groupwise nr_group={num_expert} m={m} n={n} k={k}\n"
            f"in_dtype={in_dtype} out_dtype={out_dtype}: \n"
            f"===> {tflops_per_second:.4f} TFLOPs/s\n"
            f"===> {memory_bandwidth_per_second:.4f} GB/s"
        )
        print()

    return tflops_per_second, memory_bandwidth_per_second


def bench_k_grouped_contig_bf16_tn_groupwise(
    ks_per_group,
    m,
    n,
    in_dtype,
    out_dtype,
    verbose=True,
):
    k = sum(ks_per_group)
    num_expert = len(ks_per_group)

    a_fp32 = torch.rand((k, m), device="musa", dtype=torch.float)
    b_fp32 = torch.rand((k, n), device="musa", dtype=torch.float)
    group_k_idx = torch.tensor(ks_per_group, device="musa", dtype=torch.int32)

    d = torch.zeros((num_expert, m, n), device="musa", dtype=out_dtype)

    a_bf16 = a_fp32.clone().to(in_dtype)
    b_bf16 = b_fp32.clone().to(in_dtype)

    measurements = bench_gpu_time(
        lambda: k_grouped_bf16_gemm_tn_contiguous(
            a_bf16,
            b_bf16,
            d,
            None,
            group_k_idx,
        ),
        dry_run_time_ms=100,
        repeat_time_ms=1000,
        l2_flush=True,
    )
    ms = np.median(measurements)
    tflops_per_second = 2 * m * n * k * 1e-9 / ms
    total_bytes = m * k + n * k + num_expert * m * n * 2 * 4
    total_bytes += (group_k_idx.numel()) * 4
    memory_bandwidth_per_second = total_bytes * 1e-6 / ms

    if verbose:
        print(
            f"bench_k_grouped_contig_bf16_tn_groupwise nr_group={num_expert} m={m} n={n} k={k}\n"
            f"in_dtype={in_dtype} out_dtype={out_dtype}: \n"
            f"===> {tflops_per_second:.4f} TFLOPs/s\n"
            f"===> {memory_bandwidth_per_second:.4f} GB/s"
        )
        print()

    return tflops_per_second, memory_bandwidth_per_second


def get_group_deepgemm_fp8_nt_groupwise_cases():
    # [nr_group, m_per_group, n, k, select_tile_m]
    # valid select_tile_m: 128(default), 256
    return [
        (4, 8192, 4096, 7168, 128),
        (4, 8192, 4096, 7168, 256),
        (4, 8192, 7168, 2048, 128),
        (4, 8192, 7168, 2048, 256),
        (8, 4096, 4096, 7168, 128),
        (8, 4096, 4096, 7168, 256),
        (8, 4096, 7168, 2048, 128),
        (8, 4096, 7168, 2048, 256),
        (32, 256, 4096, 7168, 128),
        (32, 256, 4096, 7168, 256),
        (32, 256, 7168, 2048, 128),
        (32, 256, 7168, 2048, 256),
        (256, 128, 512, 7168, 128),
        (256, 128, 7168, 512, 128),
    ]


def get_batch_deepgemm_fp8_nt_groupwise_cases():
    # [nr_group, max_m_per_group, expect_m_per_group, n, k, select_tile_m]
    # valid select_tile_m: 0(default), 128, 256
    cases = []
    cases.extend(
        [
            (nr_group, 4096, expect_m_per_group, 4096, 7168, select_tile_m)
            for nr_group in [4, 8, 16]
            for expect_m_per_group in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            for select_tile_m in [128, 256]
        ]
    )

    cases.extend(
        [
            (nr_group, 4096, expect_m_per_group, 7168, 2048, select_tile_m)
            for nr_group in [4, 8, 16]
            for expect_m_per_group in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
            for select_tile_m in [128, 256]
        ]
    )

    return cases


def k_grouped_contig_cases():
    k_list = [
        [
            512,
            768,
            640,
            384,
            1152,
            1536,
            640,
            640,
            640,
            768,
            896,
            640,
            640,
            1152,
            896,
            896,
            1280,
            768,
            640,
            640,
            1152,
            512,
            384,
            768,
            768,
            640,
            896,
            640,
            1152,
            768,
            512,
            1152,
            768,
            512,
            512,
            640,
            512,
            768,
            512,
            896,
            768,
            896,
            768,
            640,
            896,
            768,
            896,
            896,
        ],
        [
            403,
            649,
            628,
            338,
            1095,
            1426,
            589,
            591,
            626,
            709,
            871,
            542,
            586,
            1072,
            797,
            811,
            1176,
            658,
            611,
            604,
            1113,
            505,
            266,
            741,
            647,
            587,
            847,
            628,
            1051,
            764,
            436,
            1047,
            707,
            476,
            453,
            535,
            459,
            687,
            482,
            808,
            727,
            803,
            738,
            583,
            854,
            690,
            863,
            886,
        ],
    ]
    mn_list = [(2048, 7168), (7168, 2048), (4096, 7168), (7168, 4096)]

    return [(m, n, ks) for (m, n) in mn_list for ks in k_list]


if __name__ == "__main__":
    print("=== DeepGEMM Grouped FP8 GEMM Benchmark ===\n")
    mn = [(2048, 7168), (7168, 2048), (4096, 7168), (7168, 4096)]
    for m, n, ks in k_grouped_contig_cases():
        bench_k_grouped_contig_fp8_tn_groupwise(
            ks,
            m,
            n,
            torch.float8_e4m3fn,
            torch.float,
            verbose=True,
        )
    for m, n, ks in k_grouped_contig_cases():
        bench_k_grouped_contig_bf16_tn_groupwise(
            ks,
            m,
            n,
            torch.bfloat16,
            torch.float,
            verbose=True,
        )

    for (
        nr_group,
        m,
        n,
        k,
        select_tile_m,
    ) in get_group_deepgemm_fp8_nt_groupwise_cases():
        bench_group_deepgemm_fp8_nt_groupwise(
            nr_group, m, n, k, select_tile_m, torch.float8_e4m3fn, torch.bfloat16
        )

    for (
        nr_group,
        max_m_per_group,
        expect_m_per_group,
        n,
        k,
        select_tile_m,
    ) in get_batch_deepgemm_fp8_nt_groupwise_cases():
        bench_batch_deepgemm_fp8_nt_groupwise(
            nr_group,
            max_m_per_group,
            n,
            k,
            expect_m_per_group,
            select_tile_m,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
