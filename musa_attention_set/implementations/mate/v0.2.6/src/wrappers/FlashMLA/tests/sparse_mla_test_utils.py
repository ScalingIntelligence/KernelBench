from __future__ import annotations

import enum
from typing import Optional, Tuple, List

import torch


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

    anomaly_checks = [
        deal_with_anomalies(float("inf")),
        deal_with_anomalies(float("-inf")),
        deal_with_anomalies(float("nan")),
    ]
    anomalies_check_passed = all(anomaly_checks)

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


def _ref_sparse_mla_prefill_v3_features(
    q, kv, indices, sm_scale, topk_length=None, attn_sink=None, d_v=512
):
    q = q.float()
    kv = kv.float()
    sq, h, dim_q = q.shape
    sk, g, _ = kv.shape
    dim = d_v
    h_per_group = h // g

    valid_mask = (indices >= 0) & (indices < sk)
    if topk_length is not None:
        pos = torch.arange(indices.shape[-1], device=indices.device).view(1, 1, -1)
        valid_mask = valid_mask & (pos < topk_length.view(sq, 1, 1))

    safe_indices = torch.where(valid_mask, indices, indices.new_zeros(()))
    kv_by_group = kv.permute(1, 0, 2).contiguous()
    group_idx = torch.arange(g, device=q.device)[:, None, None]
    selected_kv = kv_by_group[group_idx, safe_indices.permute(1, 0, 2).long()]

    q_grouped = q.view(sq, g, h_per_group, dim_q).permute(1, 2, 0, 3).contiguous()
    logits = torch.einsum("ghsd,gstd->ghst", q_grouped, selected_kv) * sm_scale
    valid_group = valid_mask.permute(1, 0, 2).unsqueeze(1)
    logits = logits.masked_fill(~valid_group, float("-inf"))

    lse_grouped = torch.logsumexp(logits, dim=-1)
    max_grouped = logits.max(dim=-1).values
    no_valid = ~valid_group.any(dim=-1).expand_as(lse_grouped)
    lse_grouped = torch.where(
        no_valid, torch.full_like(lse_grouped, float("inf")), lse_grouped
    )

    probs = torch.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out_grouped = torch.einsum("ghst,gstd->gshd", probs, selected_kv[..., :dim])
    out = out_grouped.permute(1, 0, 2, 3).reshape(sq, h, dim)
    lse = lse_grouped.permute(2, 0, 1).reshape(sq, h)
    max_logits = max_grouped.permute(2, 0, 1).reshape(sq, h)

    if attn_sink is not None:
        out *= torch.sigmoid(lse - attn_sink.view(1, h)).unsqueeze(-1)
    out = torch.where(
        no_valid.permute(2, 0, 1).reshape(sq, h, 1), torch.zeros_like(out), out
    )
    return out.to(torch.bfloat16), max_logits, lse


def _expected_flashmla_num_splits(
    *,
    topk: int,
    batch_size: int,
    num_mp_parts: int,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk: int = 0,
    extra_topk_length: Optional[torch.Tensor] = None,
    block_size_n: int = 64,
    fixed_overhead_num_blocks: int = 5,
) -> list[int]:
    num_blocks_per_batch = []
    for batch_idx in range(batch_size):
        if topk_length is not None:
            cur_s_k = max(int(topk_length[batch_idx].item()), 0)
            if cur_s_k == 0:
                cur_s_k = 1
            if extra_topk > 0:
                cur_s_k = ((cur_s_k + block_size_n - 1) // block_size_n) * block_size_n
                if extra_topk_length is not None:
                    cur_s_k += max(int(extra_topk_length[batch_idx].item()), 0)
                else:
                    cur_s_k += extra_topk
        else:
            cur_s_k = topk
            if extra_topk_length is not None:
                cur_s_k = (
                    (max(cur_s_k, 1) + block_size_n - 1) // block_size_n
                ) * block_size_n
                cur_s_k += max(int(extra_topk_length[batch_idx].item()), 0)
            elif extra_topk > 0:
                cur_s_k += extra_topk
        last_token_idx = max(cur_s_k - 1, 0)
        num_blocks_per_batch.append(last_token_idx // block_size_n + 1)

    total_num_blocks = sum(
        num_blocks + fixed_overhead_num_blocks for num_blocks in num_blocks_per_batch
    )
    payload = (total_num_blocks + num_mp_parts - 1) // num_mp_parts
    payload += fixed_overhead_num_blocks

    expected = [0]
    now_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    for _ in range(num_mp_parts):
        remain_payload = payload
        while now_idx < batch_size:
            num_blocks = num_blocks_per_batch[now_idx]
            now_remain_blocks = num_blocks - now_block
            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                cum_num_splits += now_n_split_idx + 1
                expected.append(cum_num_splits)
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks
                now_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - fixed_overhead_num_blocks > 0:
                    now_block += remain_payload - fixed_overhead_num_blocks
                    now_n_split_idx += 1
                    remain_payload = 0
                break
    return expected[: batch_size + 1]


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
