# ruff: noqa
import pdb  # noqa: F401
import time  # noqa: F401
import math  # noqa: F401
import torch
import pytest
from typing import Optional, Tuple  # noqa: F401
import random
import enum
from typing import List
import mate
from mate.testing import supported_musa_compute_capability
from mate.execution_context import (
    maybe_fake_tensor_mode,
    is_dry_run_enabled,
    empty_if_dry_run,
)

from mate.testing.sparse_mla import (
    ref_sparse_mla_fwd_interface as ref_sparse_mla_fwd_interface_model1,
)


USE_FAKE_MODE = is_dry_run_enabled()


def build_model1_kv_cache(kv_bf16: torch.Tensor) -> torch.Tensor:
    """Quantize bf16 KV to FlashMLA MODEL1 official 584-byte layout."""
    assert kv_bf16.ndim == 3 and kv_bf16.shape[-1] == 512
    skv, hkv, d = kv_bf16.shape
    assert d == 512
    result = torch.empty((skv, hkv, 584), dtype=torch.uint8, device=kv_bf16.device)
    nope_fp8 = result[:, :, :448].view(torch.float8_e4m3fn)
    rope_bf16 = result[:, :, 448:576].view(torch.bfloat16)
    scale_bytes = result[:, :, 576:584]
    rope_bf16[:] = kv_bf16[:, :, 448:]
    scale_bytes[:, :, 7] = 0

    nope = kv_bf16[:, :, :448].float()
    for tile_idx in range(7):
        tile = nope[:, :, tile_idx * 64 : (tile_idx + 1) * 64]
        scale_inv = tile.abs().max(dim=-1).values / 448.0
        scale_inv = torch.pow(2, torch.clamp_min(scale_inv, 1e-4).log2().ceil())
        scale_bytes[:, :, tile_idx] = (torch.log2(scale_inv) + 127).to(torch.uint8)
        nope_fp8[:, :, tile_idx * 64 : (tile_idx + 1) * 64] = (
            tile / scale_inv.unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
    return result.contiguous()


def dequant_kv(kv_b):
    """Dequant official 584-byte MODEL1 FP8 KV cache to bf16.

    Layout:
      bytes 0-447   : fp8_e4m3 NoPE (448 values, 7 tiles x 64 dims)
      bytes 448-575 : bf16 RoPE (64 values)
      bytes 576-583 : fp8_e8m0 scales (7 valid + 1 padding)
    """
    skv, hkv, b = kv_b.shape
    if b != 584:
        raise ValueError(f"Unsupported KV cache layout: {b} bytes (expected 584)")
    nope_fp8 = kv_b[:, :, :448].view(torch.float8_e4m3fn)
    rope_bf16 = kv_b[:, :, 448 : 448 + 128].view(torch.bfloat16)
    scales_uint8 = kv_b[:, :, 576 : 576 + 8]
    # Manual bit-cast e8m0 -> float32:
    #   e8m0: E8M0 format, bias=127, value = 2^(byte-127)
    #   For power-of-2, float32_bits = byte << 23  (sign=0, exp=byte, mantissa=0)
    fp32_bits = scales_uint8[:, :, :7].to(torch.int32) << 23
    scales_float = fp32_bits.view(torch.float32)
    scales_expanded = scales_float.repeat_interleave(64, dim=-1)
    nope_dequant = (nope_fp8.to(torch.float32) * scales_expanded).to(torch.bfloat16)
    return torch.cat([nope_dequant, rope_bf16], dim=-1).contiguous()


# torch.random.manual_seed(42)


class FP8KVCacheLayout(enum.Enum):
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7),
        }[self]


def _cast_scale_inv_to_ue8m0(
    scales_inv: torch.Tensor, out_dtype=torch.float32
) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)


def quantize_k_cache(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
        result = torch.empty(
            (num_blocks, block_size + 1, bytes_per_token),
            dtype=torch.float8_e4m3fn,
            device=input_k_cache.device,
        )[:, :block_size, :]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        result_k_rope_part = result[..., d_nope + num_tiles * 4 :].view(
            input_k_cache.dtype
        )
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(torch.float8_e4m3fn)
            result_k_nope_part[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        bytes_per_token = d_nope + 2 * d_rope + num_tiles + 1
        size_per_block_padded = (block_size * bytes_per_token + 576 - 1) // 576 * 576
        result = torch.empty(
            (num_blocks, size_per_block_padded),
            dtype=torch.float8_e4m3fn,
            device=input_k_cache.device,
        )[:, : block_size * bytes_per_token]
        result_k_nope_rope_part = result[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        result_k_nope = result_k_nope_rope_part[
            :, :, :d_nope
        ]  # [num_blocks, block_size, d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(
            input_k_cache.dtype
        )  # [num_blocks, block_size, d_rope]
        result_k_scale_factor = (
            result[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.uint8)
        )

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / 448.0
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = (
                torch.log2(cur_scale_factors_inv).to(torch.int32) + 127
            ).to(torch.uint8)

            cur_scale_factors_inv = cur_scale_factors_inv.view(
                num_blocks, block_size, 1
            )
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(torch.float8_e4m3fn)
            result_k_nope[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_quantized_nope
            )

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        input_rope = quant_k_cache[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]
        input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        input_nope = input_nope_rope[:, :, :d_nope]
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = (
            quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.uint8)
        )

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.bfloat16)
            cur_scale_bits = input_scale[:, :, tile_idx].to(torch.int32) << 23
            cur_scales = (
                cur_scale_bits.view(torch.float32).to(torch.bfloat16).unsqueeze(-1)
            )
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")

    result = result.view(num_blocks, block_size, 1, d)
    return result


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:

    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(
        0,
        (
            abs_indices // block_size
            + torch.arange(0, b).view(b, 1, 1) * max_blocks_per_seq
        ).view(-1),
    )
    indices_in_kvcache = (
        real_block_idxs.view(b, s_q, topk) * block_size + abs_indices % block_size
    )
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache


def check_is_bitwise_equal_comparator(
    ans: torch.Tensor, ref: torch.Tensor, result: torch.Tensor
):
    """
    Return if two tensors are bitwise equal
    Return a bool if avoid_sync is False, else return a tensor
    """
    assert ans.shape == ref.shape, "Shape mismatch"
    torch.all(torch.eq(ans, ref), out=result)


def check_is_bitwise_equal(
    name: str, ans: torch.Tensor, ref: torch.Tensor, quiet: bool = False
) -> bool:
    is_bitwise_equal = torch.equal(ans, ref)
    if not quiet and not is_bitwise_equal:
        print(
            f"`{name}` mismatch: not bitwise equal. Mismatch count: {(ans != ref).sum().item()} out of {ans.numel()}"
        )
    return is_bitwise_equal


def get_cos_diff(ans: torch.Tensor, ref: torch.Tensor) -> float:
    """
    Calculate the cosine diff between two tensors
    Return a float if avoid_sync is False, else return a tensor
    """
    ans, ref = ans.double(), ref.double()
    if (ref * ref).sum().item() < 1e-12:
        return 0
    denominator = (ans * ans + ref * ref).sum().item()
    sim = 2 * (ans * ref).sum().item() / denominator
    return 1 - sim


def check_is_allclose(
    name: str,
    ans: torch.Tensor,
    ref: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
    quiet: bool = False,
) -> bool:
    """
    Check if two tensors are close enough
    Return a bool if avoid_sync is False, else return a tensor
    """
    assert ans.shape == ref.shape, (
        f"`{name}` Shape mismatch: {ans.shape} vs {ref.shape}"
    )
    assert ans.dtype == ref.dtype, (
        f"`{name}` Dtype mismatch: {ans.dtype} vs {ref.dtype}"
    )

    ans = ans.clone().to(torch.float)
    ref = ref.clone().to(torch.float)

    def report_err(*args, **kwargs):
        if not quiet:
            print(*args, **kwargs)

    # Deal with anomalies
    def deal_with_anomalies(val: float):
        ref_mask = (ref == val) if (val == val) else (ref != ref)
        ans_mask = (ans == val) if (val == val) else (ans != ans)
        ref[ref_mask] = 0.0
        ans[ans_mask] = 0.0
        if not torch.equal(ref_mask, ans_mask):
            report_err(
                f"`{name}` Anomaly number `{val}` mismatch: {ans_mask.sum().item()} in ans but {ref_mask.sum().item()} in ref"
            )
            return False
        return True

    anomalies_check_passed = True
    anomalies_check_passed &= deal_with_anomalies(float("inf"))
    anomalies_check_passed &= deal_with_anomalies(float("-inf"))
    anomalies_check_passed &= deal_with_anomalies(float("nan"))

    cos_diff = get_cos_diff(ans, ref)
    raw_abs_err = torch.abs(ans - ref)
    raw_rel_err = raw_abs_err / (torch.abs(ref) + (1e-6))
    rel_err = raw_rel_err.masked_fill(raw_abs_err < abs_tol, 0)
    abs_err = raw_abs_err.masked_fill(raw_rel_err < rel_tol, 0)
    pass_mask = (abs_err < abs_tol) | (rel_err < rel_tol)

    if not anomalies_check_passed:
        return False

    if not pass_mask.all():
        report_err(f"`{name}` mismatch")
        max_abs_err_pos: int = torch.argmax(abs_err, keepdim=True).item()
        max_rel_err_pos: int = torch.argmax(rel_err, keepdim=True).item()

        def get_pos_in_tensor(t: torch.Tensor, pos: int) -> List[int]:
            result = []
            for size in t.shape[::-1]:
                result.append(pos % size)
                pos = pos // size
            assert pos == 0
            return result[::-1]

        report_err(
            f"max abs err: {torch.max(abs_err).item()}: pos {get_pos_in_tensor(ans, max_abs_err_pos)}, {ans.reshape(-1)[max_abs_err_pos].item()} vs {ref.reshape(-1)[max_abs_err_pos].item()}"
        )
        report_err(
            f"max rel err: {torch.max(rel_err).item()}: pos {get_pos_in_tensor(ans, max_rel_err_pos)}, {ans.reshape(-1)[max_rel_err_pos].item()} vs {ref.reshape(-1)[max_rel_err_pos].item()}"
        )
        report_err(
            f"{pass_mask.sum()} out of {pass_mask.numel()} passed ({pass_mask.sum() / pass_mask.numel() * 100.0:.2f}%)"
        )
        report_err(f"Cosine diff: {cos_diff} (threshold: {cos_diff_tol})")
        return False
    else:
        if abs(cos_diff) > cos_diff_tol:
            report_err(
                f"`{name}` mismatch: Cosine diff too large: {cos_diff} vs {cos_diff_tol})"
            )
            return False
        return True


def check_is_allclose_comparator(
    name: str,
    ans: torch.Tensor,
    ref: torch.Tensor,
    out: torch.Tensor,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-2,
    cos_diff_tol: float = 1e-7,
):
    out.fill_(check_is_allclose(name, ans, ref, abs_tol, rel_tol, cos_diff_tol))


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


def ref_sparse_mla_fwd_interface(q, kv, indices, sm_scale=None, is_casual=True):
    q = q.float()
    kv = kv.float()
    indices = indices.transpose(0, 1).long()
    sq, h, dim_q = q.shape
    sk, g, _ = kv.shape

    # assert kv.shape[-1] == 576, "you should assign dim otherwise"
    dim = 512
    v = kv[..., :dim]

    _, _, dim_v = v.shape
    g_index = g
    h_index = h // g
    valid_mask = (indices >= 0) & (indices < sk)
    indices_clamped = torch.where(valid_mask, indices, indices.new_full((), sk))

    kv_by_group = kv.permute(1, 0, 2).contiguous()
    v_by_group = v.permute(1, 0, 2).contiguous()
    kv_with_padding = torch.cat(
        [kv_by_group, kv_by_group.new_zeros(g_index, 1, kv.shape[-1])], dim=1
    )
    v_with_padding = torch.cat(
        [v_by_group, v_by_group.new_zeros(g_index, 1, dim_v)], dim=1
    )

    group_idx = torch.arange(g_index, device=q.device)[:, None, None]
    k_selected = kv_with_padding[group_idx, indices_clamped]
    v_selected = v_with_padding[group_idx, indices_clamped]

    q = q.view(sq, g, -1, dim_q).permute(1, 2, 0, 3).contiguous()
    score = torch.einsum("ghmd,gmnd->ghmn", q, k_selected)
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
    score = score.mul(sm_scale)
    score = score.masked_fill(~valid_mask[:, None, :, :], float("-inf"))
    p = score.softmax(dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    p = p.view(g_index, h_index, sq, indices.shape[-1])
    p = p.view(g, -1, sq, indices.shape[-1])
    o = torch.einsum("ghmn,gmnd->mghd", p.type(v_selected.dtype), v_selected)
    o = o.reshape(sq, h, dim_v)
    return o.to(torch.bfloat16), score


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("batch", [1, 2, 4, 8])
@pytest.mark.parametrize("sq", [1, 2, 3, 8])
@pytest.mark.parametrize("skv", [65536, 1024])
@pytest.mark.parametrize(
    "heads",
    [
        128,
        64,
    ],
)
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "sm_scale",
    [
        0.1352337788608801,
        0.0625,
    ],
)
def test_dsa_decode(batch, sq, skv, heads, hkv, dqk, dv, topk, dtype, sm_scale):
    device = get_test_device()
    q = (
        torch.randn((batch, sq, heads, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )
    pagesize = 64
    pagenum = (skv + pagesize - 1) // pagesize

    kv = (
        torch.randn((pagenum, pagesize, hkv, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((batch, sq, hkv, topk), -1, dtype=torch.int32, device=device)
    for b in range(batch):
        for t in range(sq):
            if random.random() < 0.8:
                for h in range(hkv):
                    i_i = torch.randperm(skv, device=device)[:topk]
                    indices[b, t, h, : len(i_i)] = i_i
    kcache = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = dequantize_k_cache(kcache, FP8KVCacheLayout.V32_FP8Sparse).contiguous()

    tile_scheduler_metadata, num_splits = empty_if_dry_run(
        mate.flashmla.get_mla_metadata,
        last=False,
        empty_values=[
            torch.empty(
                (
                    64,
                    8,
                ),
                dtype=torch.int32,
                device=device,
            ),
            torch.empty((batch + 1,), dtype=torch.int32, device=device),
        ],
    )(
        cache_seqlens=None,
        num_q_tokens_per_head_k=sq * heads // 1,
        num_heads_k=1,
        num_heads_q=heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
    )

    # import pdb

    # pdb.set_trace()

    tl_out, lse = mate.flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=kcache.view(pagenum, pagesize, hkv, 656),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=dv,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
    )
    torch.musa.synchronize()

    ref_out, _ = ref_sparse_mla_fwd_interface(
        q.view(batch * sq, heads, dqk),
        kv_dequant.view(pagenum * pagesize, hkv, -1),
        indices.view(batch * sq, hkv, topk),
        sm_scale=sm_scale,
    )
    is_out_correct = check_is_allclose(
        "output",
        tl_out.view(-1),
        ref_out.to(device).view(-1),
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert is_out_correct


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("sq", [128, 256, 255])
@pytest.mark.parametrize("skv", [65536, 32768, 1024])
@pytest.mark.parametrize(
    "heads",
    [
        128,
        64,
    ],
)
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize(
    "sm_scale",
    [
        0.1352337788608801,
        0.0625,
    ],
)
def test_dsa_prefill(sq, skv, heads, hkv, dqk, dv, topk, dtype, sm_scale):
    device = get_test_device()
    q = (
        torch.randn((sq, heads, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )
    kv = (
        torch.randn((skv, hkv, dqk), dtype=dtype, device=device) / 10
        + (random.random() - 0.5) / 10
    )

    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((sq, hkv, topk), -1, dtype=torch.int32, device=device)
    for t in range(sq):
        if random.random() < 0.8:
            for h in range(hkv):
                i_i = torch.randperm(skv, device=device)[:topk]
                indices[t, h, : len(i_i)] = i_i

    tl_out, _, _ = mate.flashmla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=dv,
        attn_sink=None,
        topk_length=None,
    )
    torch.musa.synchronize()

    ref_out, _ = ref_sparse_mla_fwd_interface(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
    )
    is_out_correct = check_is_allclose(
        "output",
        tl_out.view(-1),
        ref_out.to(device).view(-1),
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert is_out_correct


MODEL1_PREFILL_CASES = [
    # (tag, S, SKV, topk, H, topk_len, sink, mostly_invalid, all_invalid, future_indices)
    ("basic_small", 1, 128, 128, 64, False, False, False, False, False),
    ("basic_oob", 213, 95, 128, 128, False, False, False, False, False),
    ("dynamic_len", 321, 512, 128, 128, True, False, False, False, False),
    ("attn_sink", 213, 153, 256, 64, True, True, False, False, False),
    ("corner_many_oob", 1024, 1024, 2048, 64, False, False, False, False, False),
    ("all_invalid", 321, 512, 128, 128, True, True, False, True, False),
]


MODEL1_PREFILL_RANDOM_SHAPE_CASES = [
    # (S, SKV). Keep SKV coverage small, and vary S over random 1K-8K values.
    (1818, 1990),
    (1876, 4755),
    (3239, 7454),
    (3314, 8012),
    (3652, 1990),
    (4460, 4755),
    (5794, 7454),
    (6015, 8012),
    (6146, 1990),
    (6255, 4755),
    (7514, 7454),
    (7756, 8012),
]


MODEL1_PREFILL_SELF_COMPARE_STRESS_CASES = [
    # (S, SKV). Compare prefill against itself under a query-row permutation.
    (1876, 4755),
    (3314, 8012),
    (6015, 8012),
]


@pytest.mark.parametrize(
    "seq_len,num_heads,have_topk_length,expected_persistence",
    [
        (65535, 64, True, True),
        (65536, 64, True, False),
        (70000, 64, False, True),
        (70000, 128, True, True),
    ],
)
def test_model1_sparse_prefill_persistence_selector(
    seq_len,
    num_heads,
    have_topk_length,
    expected_persistence,
):
    from mate.sparse_mla.tilelang import sparse_mla_prefill

    topk_length = (
        torch.empty((seq_len,), dtype=torch.int32, device="meta")
        if have_topk_length
        else None
    )

    assert (
        sparse_mla_prefill._use_model1_persistence(
            seq_len,
            num_heads,
            topk_length,
        )
        is expected_persistence
    )


def test_v32_sparse_prefill_dispatch_ignores_model1_persistence_selector(monkeypatch):
    from mate.sparse_mla.tilelang import sparse_mla_prefill

    calls = []

    def fake_v32(*args, **kwargs):
        calls.append((args, kwargs))
        return object(), object(), object()

    monkeypatch.setattr(sparse_mla_prefill, "_prefill_v32", fake_v32)
    seq_len = 70000
    q = torch.empty((seq_len, 64, 576), dtype=torch.bfloat16, device="meta")
    kv = torch.empty((256, 1, 576), dtype=torch.bfloat16, device="meta")
    indices = torch.empty((seq_len, 1, 256), dtype=torch.int32, device="meta")
    topk_length = torch.empty((seq_len,), dtype=torch.int32, device="meta")

    sparse_mla_prefill.sparse_mla_prefill_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=576**-0.5,
        topk_length=topk_length,
    )

    assert len(calls) == 1
    assert "is_persistence" not in calls[0][1]


MODEL1_DECODE_CASES = [
    # (tag, B, S, SKV, topk, SKV_EXTRA, extra_topk, H, topk_len, extra_topk_len, sink)
    ("basic", 128, 1, 8192, 2048, 0, 0, 128, False, False, False),
    ("basic_small", 4, 1, 512, 64, 0, 0, 64, False, False, False),
    ("extra", 74, 1, 1024, 576, 1024, 576, 128, False, False, False),
    ("dynamic_len", 321, 1, 2046, 2048, 2046, 2048, 128, True, True, False),
    ("attn_sink", 32, 1, 1024, 576, 1024, 576, 128, True, True, True),
    ("all_invalid", 32, 1, 512, 64, 512, 64, 64, True, True, True),
]


MODEL1_DECODE_RANDOM_SHAPE_CASES = [
    # (B, S, SKV). Keep SKV coverage small, and vary total decode queries.
    (1818, 1, 1990),
    (1876, 1, 4755),
    (3239, 1, 7454),
    (3314, 1, 8012),
    (3652, 1, 1990),
    (4460, 1, 4755),
    (5794, 1, 7454),
    (6015, 1, 8012),
    (6146, 1, 1990),
    (6255, 1, 4755),
    (7514, 1, 7454),
    (7756, 1, 8012),
]


def test_model1_sparse_prefill_seq_pack_dispatch(monkeypatch):
    from mate.sparse_mla import flashmla_sparse

    calls = []

    def fake_pack_interface(**kwargs):
        calls.append(kwargs)
        q = kwargs["q"]
        return (
            torch.empty_like(q),
            torch.empty(q.shape[:2], dtype=torch.float32),
            torch.empty(q.shape[:2], dtype=torch.float32),
        )

    monkeypatch.setattr(
        flashmla_sparse,
        "sparse_mla_fwd_interface_model1_pack",
        fake_pack_interface,
    )

    for heads, token_pack in ((8, 8), (16, 4), (32, 2)):
        calls.clear()
        seq_len = token_pack * 2
        q = torch.empty((seq_len, heads, 512), dtype=torch.bfloat16)
        kv = torch.empty((seq_len, 1, 512), dtype=torch.bfloat16)
        indices = (
            torch.arange(seq_len * 64, dtype=torch.int32)
            .reshape(seq_len, 1, 64)
            .contiguous()
        )
        topk_length = torch.arange(seq_len, dtype=torch.int32)

        out = flashmla_sparse._try_model1_seq_pack_sparse_prefill(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=512**-0.5,
            d_v=512,
            attn_sink=None,
            topk_length=topk_length,
        )

        assert out is not None
        assert len(calls) == 1
        call = calls[0]
        assert call["token_pack"] == token_pack
        assert call["compressed_kv_len"] == 0
        assert call["compress_ratio"] == 1
        torch.testing.assert_close(
            call["indices"],
            indices[token_pack - 1 :: token_pack].contiguous(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            call["topk_length"],
            topk_length[token_pack - 1 :: token_pack].contiguous(),
            rtol=0,
            atol=0,
        )


def _ref_sparse_mla_decode_model1(
    q,
    kv,
    indices,
    extra_kv=None,
    extra_indices=None,
    topk_length=None,
    extra_topk_length=None,
    sm_scale=None,
    attn_sink=None,
    d_v=512,
):
    q = q.float()
    kv = kv.float()
    sq, h, dim_q = q.shape
    sk, _, _ = kv.shape
    dim = d_v
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale

    indices_orig = indices.clone()
    if topk_length is not None:
        pos = torch.arange(indices.shape[-1], device=indices.device).view(1, 1, -1)
        indices_orig = torch.where(
            pos < topk_length.view(-1, 1, 1),
            indices_orig,
            torch.full_like(indices_orig, -1),
        )

    invalid_mask_orig = (indices_orig < 0) | (indices_orig >= sk)
    indices_orig_safe = indices_orig.clone()
    indices_orig_safe[invalid_mask_orig] = 0
    gathered_kv_orig = kv.index_select(
        dim=0, index=indices_orig_safe.flatten()
    ).reshape(sq, -1, dim)

    if extra_kv is not None:
        extra_kv = extra_kv.float()
        sk_extra = extra_kv.shape[0]
        indices_extra = extra_indices.clone()
        if extra_topk_length is not None:
            pos = torch.arange(
                indices_extra.shape[-1], device=indices_extra.device
            ).view(1, 1, -1)
            indices_extra = torch.where(
                pos < extra_topk_length.view(-1, 1, 1),
                indices_extra,
                torch.full_like(indices_extra, -1),
            )

        invalid_mask_extra = (indices_extra < 0) | (indices_extra >= sk_extra)
        indices_extra_safe = indices_extra.clone()
        indices_extra_safe[invalid_mask_extra] = 0
        gathered_kv_extra = extra_kv.index_select(
            dim=0, index=indices_extra_safe.flatten()
        ).reshape(sq, -1, dim)
        gathered_kv = torch.cat([gathered_kv_orig, gathered_kv_extra], dim=1)
        invalid_mask = torch.cat([invalid_mask_orig, invalid_mask_extra], dim=2)
    else:
        gathered_kv = gathered_kv_orig
        invalid_mask = invalid_mask_orig

    logits = q @ gathered_kv.transpose(1, 2)
    logits *= sm_scale
    logits = logits.masked_fill(invalid_mask.view(sq, 1, -1), float("-inf"))

    lonely_q_mask = invalid_mask.view(sq, -1).all(dim=1).view(sq, 1).expand(sq, h)
    lse = torch.full((sq, h), float("+inf"), dtype=logits.dtype, device=logits.device)
    valid_q_mask = ~lonely_q_mask
    lse[valid_q_mask] = torch.logsumexp(logits[valid_q_mask], dim=-1)

    attn = torch.zeros_like(logits)
    attn[valid_q_mask] = torch.exp(
        logits[valid_q_mask] - lse[valid_q_mask].unsqueeze(-1)
    )
    out = attn @ gathered_kv[..., :dim]
    if attn_sink is not None:
        out *= torch.sigmoid(lse - attn_sink.view(1, h)).unsqueeze(-1)
    out[lonely_q_mask.unsqueeze(-1).expand_as(out)] = 0.0
    return out.to(torch.bfloat16), lse


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize(
    "case", MODEL1_PREFILL_CASES, ids=[case[0] for case in MODEL1_PREFILL_CASES]
)
def test_model1_sparse_mla_prefill(case):
    (
        tag,
        seq_len,
        seq_len_kv,
        topk,
        num_heads,
        have_topk_length,
        have_attn_sink,
        mostly_invalid,
        all_indices_invalid,
        force_future_indices,
    ) = case
    del tag

    torch.random.manual_seed(0)
    device = get_test_device()
    sm_scale = 512**-0.5
    q = torch.randn((seq_len, num_heads, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((seq_len_kv, 1, 512), dtype=torch.bfloat16, device=device)

    indices = torch.full((seq_len, 1, topk), -1, dtype=torch.int32, device=device)
    for token_idx in range(seq_len):
        max_len = max(1, token_idx + 1)
        cur_indices = torch.randperm(max_len, device=device)[:topk]
        indices[token_idx, 0, : len(cur_indices)] = cur_indices
        if force_future_indices and topk > 0 and token_idx + 1 < seq_len_kv:
            indices[token_idx, 0, 0] = token_idx + 1

    if all_indices_invalid:
        indices.fill_(2147483647)
    elif mostly_invalid:
        invalid_mask = torch.rand(indices.shape, device=device) < 0.9
        indices = torch.where(
            invalid_mask, torch.full_like(indices, 2147483647), indices
        )

    topk_length = None
    if have_topk_length:
        topk_length = torch.randint(
            1, topk + 1, (seq_len,), dtype=torch.int32, device=device
        )

    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn((num_heads,), dtype=torch.float32, device=device)
        inf_mask = torch.randn((num_heads,), dtype=torch.float32, device=device)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")

    tl_out, _, tl_lse = mate.flashmla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    ref_out, ref_lse = ref_sparse_mla_fwd_interface_model1(
        q,
        kv,
        indices,
        topk_length=topk_length,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        d_v=512,
    )
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize(
    "batch_size, seq_len_q, seq_len_kv",
    MODEL1_DECODE_RANDOM_SHAPE_CASES,
    ids=[
        f"b_{batch}_sq_{sq}_skv_{skv}"
        for batch, sq, skv in MODEL1_DECODE_RANDOM_SHAPE_CASES
    ],
)
def test_model1_sparse_mla_decode_random_shape_invalid_lanes(
    batch_size, seq_len_q, seq_len_kv
):
    torch.random.manual_seed(batch_size * 10000 + seq_len_kv)
    device = get_test_device()
    total_q = batch_size * seq_len_q
    page_size = 64
    topk = 128
    num_heads = 64
    sm_scale = 0.1352337788608801
    num_pages = (seq_len_kv + page_size - 1) // page_size

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, 512), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(
        (num_pages, page_size, 1, 512), dtype=torch.bfloat16, device=device
    )
    indices = torch.randint(
        0,
        seq_len_kv,
        (batch_size, seq_len_q, 1, topk),
        dtype=torch.int32,
        device=device,
    )
    topk_length = torch.randint(
        0, topk + 1, (batch_size,), dtype=torch.int32, device=device
    )

    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse).view(
        -1, 1, 512
    )

    tile_scheduler_metadata, num_splits = empty_if_dry_run(
        mate.flashmla.get_mla_metadata,
        last=False,
        empty_values=[
            torch.empty((64, 8), dtype=torch.int32, device=device),
            torch.empty((batch_size + 1,), dtype=torch.int32, device=device),
        ],
    )(
        cache_seqlens=None,
        num_q_tokens_per_head_k=seq_len_q * num_heads,
        num_heads_k=1,
        num_heads_q=num_heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
        bs=batch_size,
        topk_length=topk_length,
    )

    tl_out, tl_lse = mate.flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
    )

    ref_out, ref_lse = _ref_sparse_mla_decode_model1(
        q.view(total_q, num_heads, 512),
        kv_dequant,
        indices.view(total_q, 1, topk),
        topk_length=topk_length.repeat_interleave(seq_len_q),
        sm_scale=sm_scale,
    )
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, 512)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize(
    "seq_len, seq_len_kv",
    MODEL1_PREFILL_RANDOM_SHAPE_CASES,
    ids=[f"sq_{sq}_skv_{skv}" for sq, skv in MODEL1_PREFILL_RANDOM_SHAPE_CASES],
)
def test_model1_sparse_mla_prefill_random_shape_jit(seq_len, seq_len_kv):
    torch.random.manual_seed(seq_len * 10000 + seq_len_kv)
    device = get_test_device()
    topk = 128
    num_heads = 64
    sm_scale = 512**-0.5

    q = torch.randn((seq_len, num_heads, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((seq_len_kv, 1, 512), dtype=torch.bfloat16, device=device)
    indices = torch.full((seq_len, 1, topk), -1, dtype=torch.int32, device=device)
    for token_idx in range(seq_len):
        max_len = min(seq_len_kv, token_idx + 1)
        cur_indices = torch.randperm(max_len, device=device)[:topk]
        indices[token_idx, 0, : len(cur_indices)] = cur_indices
    topk_length = torch.randint(
        1, topk + 1, (seq_len,), dtype=torch.int32, device=device
    )

    tl_out, _, tl_lse = mate.flashmla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=512,
        topk_length=topk_length,
    )

    ref_out, ref_lse = ref_sparse_mla_fwd_interface_model1(
        q,
        kv,
        indices,
        topk_length=topk_length,
        sm_scale=sm_scale,
        d_v=512,
    )
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize(
    "seq_len, seq_len_kv",
    MODEL1_PREFILL_SELF_COMPARE_STRESS_CASES,
    ids=[f"sq_{sq}_skv_{skv}" for sq, skv in MODEL1_PREFILL_SELF_COMPARE_STRESS_CASES],
)
def test_model1_sparse_mla_prefill_self_compare_stress(seq_len, seq_len_kv):
    torch.random.manual_seed(seq_len * 10000 + seq_len_kv)
    device = get_test_device()
    topk = 128
    num_heads = 64
    sm_scale = 512**-0.5

    q = torch.randn((seq_len, num_heads, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn((seq_len_kv, 1, 512), dtype=torch.bfloat16, device=device)
    indices = torch.full((seq_len, 1, topk), -1, dtype=torch.int32, device=device)
    for token_idx in range(seq_len):
        max_len = min(seq_len_kv, token_idx + 1)
        cur_indices = torch.randperm(max_len, device=device)[:topk]
        indices[token_idx, 0, : len(cur_indices)] = cur_indices
    topk_length = torch.randint(
        1, topk + 1, (seq_len,), dtype=torch.int32, device=device
    )

    base_out, _, base_lse = mate.flashmla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=512,
        topk_length=topk_length,
    )
    base_out = base_out.clone()
    base_lse = base_lse.clone()

    for _ in range(1000):
        cur_out, _, cur_lse = mate.flashmla.flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=sm_scale,
            d_v=512,
            topk_length=topk_length,
        )
        torch.testing.assert_close(base_out, cur_out, rtol=0, atol=0)
        torch.testing.assert_close(base_lse, cur_lse, rtol=0, atol=0)
        torch.musa.synchronize()


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize(
    "case", MODEL1_DECODE_CASES, ids=[case[0] for case in MODEL1_DECODE_CASES]
)
def test_model1_sparse_mla_decode(case):
    (
        tag,
        batch_size,
        seq_len_q,
        seq_len_kv,
        topk,
        seq_len_kv_extra,
        extra_topk,
        num_heads,
        have_topk_length,
        have_extra_topk_length,
        have_attn_sink,
    ) = case
    all_indices_invalid = tag == "all_invalid"

    torch.random.manual_seed(0)
    device = get_test_device()
    total_q = batch_size * seq_len_q
    page_size = 64
    num_pages = (seq_len_kv + page_size - 1) // page_size

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, 512), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(
        (num_pages, page_size, 1, 512), dtype=torch.bfloat16, device=device
    )

    indices = torch.full(
        (batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device
    )
    if not all_indices_invalid:
        for batch_idx in range(batch_size):
            for q_idx in range(seq_len_q):
                cur_indices = torch.randperm(seq_len_kv, device=device)[:topk]
                indices[batch_idx, q_idx, 0, : len(cur_indices)] = cur_indices

    topk_length = None
    if have_topk_length:
        if all_indices_invalid:
            topk_length = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        else:
            topk_length = torch.randint(
                1, topk + 1, (batch_size,), dtype=torch.int32, device=device
            )

    extra_k_cache = None
    extra_kv_dequant = None
    extra_indices = None
    extra_topk_length = None
    if extra_topk > 0:
        num_extra_pages = (seq_len_kv_extra + page_size - 1) // page_size
        extra_kv = torch.randn(
            (num_extra_pages, page_size, 1, 512), dtype=torch.bfloat16, device=device
        )
        extra_indices = torch.full(
            (batch_size, seq_len_q, 1, extra_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        if not all_indices_invalid:
            for batch_idx in range(batch_size):
                for q_idx in range(seq_len_q):
                    cur_indices = torch.randperm(seq_len_kv_extra, device=device)[
                        :extra_topk
                    ]
                    extra_indices[batch_idx, q_idx, 0, : len(cur_indices)] = cur_indices
        if have_extra_topk_length:
            if all_indices_invalid:
                extra_topk_length = torch.zeros(
                    (batch_size,), dtype=torch.int32, device=device
                )
            else:
                extra_topk_length = torch.randint(
                    1, extra_topk + 1, (batch_size,), dtype=torch.int32, device=device
                )
        extra_k_cache = quantize_k_cache(extra_kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
        extra_kv_dequant = dequantize_k_cache(
            extra_k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse
        ).view(-1, 1, 512)

    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn((num_heads,), dtype=torch.float32, device=device)
        inf_mask = torch.randn((num_heads,), dtype=torch.float32, device=device)
        attn_sink[inf_mask > 0.5] = float("inf")
        attn_sink[inf_mask < -0.5] = float("-inf")

    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse).view(
        -1, 1, 512
    )

    tl_out, tl_lse = mate.flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=None,
        num_splits=None,
        softmax_scale=0.1352337788608801,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )

    ref_out, ref_lse = _ref_sparse_mla_decode_model1(
        q.view(total_q, num_heads, 512),
        kv_dequant,
        indices.view(total_q, 1, topk),
        extra_kv=extra_kv_dequant,
        extra_indices=extra_indices.view(total_q, 1, extra_topk)
        if extra_indices is not None
        else None,
        topk_length=topk_length.repeat_interleave(seq_len_q)
        if topk_length is not None
        else None,
        extra_topk_length=extra_topk_length.repeat_interleave(seq_len_q)
        if extra_topk_length is not None
        else None,
        sm_scale=0.1352337788608801,
        attn_sink=attn_sink,
    )
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, 512)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


def _check_model1_sparse_mla_decode_auto_heads(num_heads):
    torch.random.manual_seed(0)
    device = get_test_device()
    batch_size = 4
    seq_len_q = 1
    seq_len_kv = 512
    topk = 64
    total_q = batch_size * seq_len_q
    page_size = 64
    num_pages = (seq_len_kv + page_size - 1) // page_size
    sm_scale = 0.1352337788608801

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, 512), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(
        (num_pages, page_size, 1, 512), dtype=torch.bfloat16, device=device
    )
    indices = torch.randint(
        0,
        seq_len_kv,
        (batch_size, seq_len_q, 1, topk),
        dtype=torch.int32,
        device=device,
    )
    topk_length = torch.randint(
        1, topk + 1, (batch_size,), dtype=torch.int32, device=device
    )

    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse).view(
        -1, 1, 512
    )

    tile_scheduler_metadata, num_splits = empty_if_dry_run(
        mate.flashmla.get_mla_metadata,
        last=False,
        empty_values=[
            torch.empty((64, 8), dtype=torch.int32, device=device),
            torch.empty((batch_size + 1,), dtype=torch.int32, device=device),
        ],
    )(
        cache_seqlens=None,
        num_q_tokens_per_head_k=seq_len_q * num_heads,
        num_heads_k=1,
        num_heads_q=num_heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
        bs=batch_size,
        topk_length=topk_length,
    )

    tl_out, tl_lse = mate.flashmla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=sm_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
    )

    ref_out, ref_lse = _ref_sparse_mla_decode_model1(
        q.view(total_q, num_heads, 512),
        kv_dequant,
        indices.view(total_q, 1, topk),
        topk_length=topk_length.repeat_interleave(seq_len_q),
        sm_scale=sm_scale,
    )
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, 512)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
def test_model1_sparse_mla_decode_heads64_default():
    _check_model1_sparse_mla_decode_auto_heads(64)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
def test_model1_sparse_mla_decode_heads16_auto_bm16():
    _check_model1_sparse_mla_decode_auto_heads(16)


def _padded_last_dim_view(tensor: torch.Tensor, padding: int) -> torch.Tensor:
    storage = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] + padding),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result = storage[..., : tensor.shape[-1]]
    result.copy_(tensor)
    assert not result.is_contiguous()
    return result


def test_sparse_mla_factories_keep_strides_runtime():
    from mate.sparse_mla.tilelang.sparse_mla_model1_decode_fwd_scheduled import (
        sparse_attention_decode_fwd_scheduled_kernel_model1,
    )
    from mate.sparse_mla.tilelang.sparse_mla_model1_fwd_pipelined import (
        sparse_attention_fwd_kernel_model1,
    )
    from mate.sparse_mla.tilelang.sparse_mla_v32_decode_fwd_scheduled import (
        sparse_attention_fwd_kernel as sparse_attention_v32_decode_kernel,
    )
    from mate.sparse_mla.tilelang.sparse_mla_v32_fwd_pipelined import (
        sparse_attention_fwd_kernel as sparse_attention_v32_prefill_kernel,
    )

    factories = (
        sparse_attention_fwd_kernel_model1,
        sparse_attention_v32_prefill_kernel,
        sparse_attention_decode_fwd_scheduled_kernel_model1,
        sparse_attention_v32_decode_kernel,
    )
    for factory in factories:
        factory_args = set(factory.func.arg_names)
        assert not [name for name in factory_args if "stride" in name]
        assert not {"page_block_bytes", "extra_page_block_bytes"} & factory_args


def test_sparse_mla_index_promotion_follows_tensor_byte_span():
    import tilelang

    from mate.sparse_mla.tilelang.sparse_mla_index_type import (
        INT32_ADDRESS_SPACE_BYTES,
        jit_for_tensor_addressing,
        needs_index_type_promotion,
        tensor_byte_span,
    )
    from mate.sparse_mla.tilelang.sparse_mla_v32_fwd_pipelined import (
        sparse_attention_fwd_kernel,
    )

    small = torch.empty_strided((2, 8), (16, 1), dtype=torch.uint8, device="meta")
    at_limit = torch.empty_strided(
        (2, 1),
        (INT32_ADDRESS_SPACE_BYTES - 1, 1),
        dtype=torch.uint8,
        device="meta",
    )
    over_limit = torch.empty_strided(
        (2, 1),
        (INT32_ADDRESS_SPACE_BYTES, 1),
        dtype=torch.uint8,
        device="meta",
    )
    small_view_of_large_storage = torch.empty(
        (INT32_ADDRESS_SPACE_BYTES + 1,), dtype=torch.uint8, device="meta"
    )[:1]

    assert tensor_byte_span(at_limit) == INT32_ADDRESS_SPACE_BYTES
    assert not needs_index_type_promotion(small, at_limit)
    assert tensor_byte_span(over_limit) == INT32_ADDRESS_SPACE_BYTES + 1
    assert needs_index_type_promotion(small, over_limit)
    assert not needs_index_type_promotion(small_view_of_large_storage)

    base = jit_for_tensor_addressing(sparse_attention_fwd_kernel, small)
    promoted = jit_for_tensor_addressing(sparse_attention_fwd_kernel, over_limit)
    assert base is sparse_attention_fwd_kernel
    assert promoted is jit_for_tensor_addressing(
        sparse_attention_fwd_kernel, over_limit
    )
    config_key = tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION
    assert base.pass_configs[config_key] is True
    assert promoted.pass_configs[config_key] is False


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("head_dim", [512, 576], ids=["model1", "v32"])
def test_sparse_mla_prefill_strided_inputs(head_dim):
    torch.random.manual_seed(0)
    device = get_test_device()
    seq_len = 4
    seq_len_kv = 128
    heads = 64
    topk = 64
    sm_scale = head_dim**-0.5
    q = torch.randn((seq_len, heads, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.randn((seq_len_kv, 1, head_dim), dtype=torch.bfloat16, device=device)
    indices = torch.full((seq_len, 1, topk), -1, dtype=torch.int32, device=device)
    for token_idx in range(seq_len):
        valid = torch.randperm(token_idx + 1, device=device)
        indices[token_idx, 0, : valid.numel()] = valid

    compact = mate.flashmla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=512,
    )
    for padding in (8, 16):
        strided = mate.flashmla.flash_mla_sparse_fwd(
            q=_padded_last_dim_view(q, padding),
            kv=_padded_last_dim_view(kv, padding),
            indices=_padded_last_dim_view(indices, padding),
            sm_scale=sm_scale,
            d_v=512,
        )

        for compact_tensor, strided_tensor in zip(compact, strided):
            torch.testing.assert_close(
                strided_tensor, compact_tensor, rtol=1e-2, atol=1e-2
            )


def _sparse_decode_metadata(q: torch.Tensor, topk: int):
    batch, seq_len, heads, _ = q.shape
    return mate.flashmla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=seq_len * heads,
        num_heads_k=1,
        num_heads_q=heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
        bs=batch,
    )


@supported_musa_compute_capability([31])
def test_v32_sparse_mla_decode_strided_inputs():
    from mate.sparse_mla.tilelang.sparse_mla_v32_decode_fwd_scheduled import (
        tilelang_flashmla_interface,
    )

    torch.random.manual_seed(0)
    device = get_test_device()
    batch = 2
    seq_len = 1
    seq_len_kv = 128
    heads = 64
    topk = 64
    page_size = 64
    q = torch.randn((batch, seq_len, heads, 576), dtype=torch.bfloat16, device=device)
    kv = torch.randn(
        (seq_len_kv // page_size, page_size, 1, 576),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.randint(
        0,
        seq_len_kv,
        (batch, seq_len, 1, topk),
        dtype=torch.int32,
        device=device,
    )
    cache = (
        quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse)
        .contiguous()
        .view(torch.uint8)
        .view(seq_len_kv, 1, 656)
    )
    metadata, num_splits = _sparse_decode_metadata(q, topk)
    kwargs = {
        "tile_scheduler_metadata": metadata,
        "num_splits": num_splits,
        "sm_scale": 576**-0.5,
        "d_v": 512,
    }
    compact = tilelang_flashmla_interface(q, cache, indices, **kwargs)
    for q_padding, cache_padding, indices_padding in ((8, 16, 8), (16, 32, 16)):
        strided = tilelang_flashmla_interface(
            _padded_last_dim_view(q, q_padding),
            _padded_last_dim_view(cache, cache_padding),
            _padded_last_dim_view(indices, indices_padding),
            **kwargs,
        )

        for compact_tensor, strided_tensor in zip(compact, strided):
            torch.testing.assert_close(
                strided_tensor, compact_tensor, rtol=1e-2, atol=1e-2
            )


@supported_musa_compute_capability([31])
def test_model1_sparse_mla_decode_strided_inputs():
    from mate.sparse_mla.tilelang.sparse_mla_model1_decode_fwd_scheduled import (
        sparse_mla_decode_fwd_scheduled_interface_model1,
    )

    torch.random.manual_seed(0)
    device = get_test_device()
    batch = 2
    seq_len = 1
    page_size = 69
    seq_len_kv = page_size * 2
    heads = 64
    topk = 64
    num_pages = seq_len_kv // page_size
    q = torch.randn((batch, seq_len, heads, 512), dtype=torch.bfloat16, device=device)
    kv = torch.randn(
        (num_pages, page_size, 1, 512), dtype=torch.bfloat16, device=device
    )
    indices = torch.randint(
        0,
        seq_len_kv,
        (batch, seq_len, 1, topk),
        dtype=torch.int32,
        device=device,
    )
    cache = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse).view(torch.uint8)
    logical_page_bytes = page_size * 584
    page_stride_bytes = cache.stride(0)
    cache_bytes = torch.as_strided(
        cache,
        (num_pages, logical_page_bytes),
        (page_stride_bytes, 1),
    )
    compact_bytes = cache_bytes.contiguous()

    def cache_views(cache_rows):
        return (
            cache_rows.view(torch.float8_e4m3fn),
            cache_rows.view(torch.bfloat16),
            cache_rows,
        )

    compact_cache = cache_views(compact_bytes)
    metadata, num_splits = _sparse_decode_metadata(q, topk)
    kwargs = {
        "tile_scheduler_metadata": metadata,
        "num_splits": num_splits,
        "sm_scale": 512**-0.5,
        "d_v": 512,
        "page_block_size": page_size,
    }
    compact = sparse_mla_decode_fwd_scheduled_interface_model1(
        q, *compact_cache, indices, **kwargs
    )
    assert compact_bytes.stride(0) % 16 == 8
    strided_rows = (
        _padded_last_dim_view(compact_bytes, 8),
        _padded_last_dim_view(compact_bytes, 16),
    )
    for padding, cache_rows in zip((8, 16), strided_rows):
        strided = sparse_mla_decode_fwd_scheduled_interface_model1(
            _padded_last_dim_view(q, padding),
            *cache_views(cache_rows),
            _padded_last_dim_view(indices, padding),
            **kwargs,
        )

        for compact_tensor, strided_tensor in zip(compact, strided):
            torch.testing.assert_close(
                strided_tensor, compact_tensor, rtol=1e-2, atol=1e-2
            )
