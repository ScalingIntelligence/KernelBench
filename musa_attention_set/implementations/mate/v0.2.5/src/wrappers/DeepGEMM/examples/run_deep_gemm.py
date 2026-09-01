import torch

import deep_gemm

from mate.testing.utils import (
    align,
    group_quantize_fp8,
)


def run_fp8_grouped_gemm_example():
    print("=" * 80)
    print("FP8 grouped GEMM example")

    torch.manual_seed(0)
    device = "musa"

    ms_per_group = [512, 768]
    n = 1024
    k = 1024
    alignment_m = deep_gemm.get_mk_alignment_for_contiguous_layout()
    out_dtype = torch.bfloat16

    quant_tile = 128
    scale_granularity_mnk = (1, quant_tile, quant_tile)

    num_expert = len(ms_per_group)
    aligned_ms = [align(m, alignment_m) for m in ms_per_group]
    m = sum(aligned_ms)

    a = torch.rand((m, k), device=device, dtype=torch.float)
    b = torch.rand((num_expert, n, k), device=device, dtype=torch.float)
    d = torch.empty((m, n), device=device, dtype=out_dtype)

    m_indices = torch.full((m,), -1, device=device, dtype=torch.int32)

    m_base = 0
    for i in range(num_expert):
        valid_m = ms_per_group[i]
        m_indices[m_base : m_base + valid_m] = i
        m_base += aligned_ms[i]

    quant_tile_shape_a = (1, quant_tile)
    quant_tile_shape_b = (1, quant_tile, quant_tile)
    scale_a_shape = (m, k // quant_tile)
    scale_b_shape = (num_expert, n // quant_tile, k // quant_tile)

    fp8_a, scale_a = group_quantize_fp8(
        a,
        scale_a_shape,
        quant_tile_shape_a,
        torch.float8_e4m3fn,
        "K",
    )
    fp8_b, scale_b = group_quantize_fp8(
        b,
        scale_b_shape,
        quant_tile_shape_b,
        torch.float8_e4m3fn,
        "K",
    )

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (fp8_a, scale_a),
        (fp8_b, scale_b),
        d,
        m_indices,
        scale_granularity_mnk,
        alignment_m=alignment_m,
    )

    torch.musa.synchronize()

    print(f"a.shape       = {tuple(a.shape)}")
    print(f"b.shape       = {tuple(b.shape)}")
    print(f"fp8_a.shape   = {tuple(fp8_a.shape)}")
    print(f"scale_a.shape = {tuple(scale_a.shape)}")
    print(f"fp8_b.shape   = {tuple(fp8_b.shape)}")
    print(f"scale_b.shape = {tuple(scale_b.shape)}")
    print(f"m_indices.shape = {tuple(m_indices.shape)}")
    print(f"out.shape     = {tuple(d.shape)}")
    print(f"out.dtype     = {d.dtype}")


def main():
    run_fp8_grouped_gemm_example()


if __name__ == "__main__":
    main()
