from __future__ import annotations

import pytest
import torch

from mate.jit.msa_ops import (
    _msa_fwd,
    gen_msa_ops_aot,
    make_msa_fwd_config,
)


_HAS_MUSA = hasattr(torch, "musa") and torch.musa.is_available()
_FP8 = getattr(torch, "float8_e4m3fn", None)
_DTYPES = [
    (torch.float16, "mutlass::half_t"),
    (torch.bfloat16, "mutlass::bfloat16_t"),
]
if _FP8 is not None:
    _DTYPES.append((_FP8, "mutlass::float_e4m3_t"))


@pytest.mark.parametrize("dtype,element", _DTYPES)
def test_msa_fwd_config_uses_the_fixed_sqmma_tile_shape(dtype, element):
    config = make_msa_fwd_config(causal=True, dtype=dtype)
    assert config["element"] == element
    assert config["head_ratio"] == 16
    assert config["head_dim"] == 128
    assert config["tile_kv"] == 128
    assert config["topk"] == 16
    assert not {"q_stages", "k_stages", "v_stages"}.intersection(config)


def test_msa_aot_specs_cover_all_forward_dtypes_and_causal_modes():
    names = {spec.name for spec in gen_msa_ops_aot()}
    expected = {
        f"msa_fwd_{dtype_name}_causal_{int(causal)}"
        for dtype_name in ("f16", "bf16", "fp8e4m3")
        for causal in (False, True)
        if dtype_name != "fp8e4m3" or _FP8 is not None
    }
    assert expected.issubset(names)
    assert "msa_sparse_topk_select" in names
    assert len(names) == len(expected) + 1


def test_msa_fwd_rejects_unsupported_sqmma_head_ratio():
    q = torch.empty((1, 4, 128), dtype=torch.float16, device="cpu")
    k = torch.empty((1, 128, 1, 128), dtype=torch.float16, device="cpu")
    v = torch.empty_like(k)
    blocks = torch.zeros((1, 1, 16), dtype=torch.int32, device="cpu")
    cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cpu")
    kv_len = torch.tensor([128], dtype=torch.int32, device="cpu")
    qo_offset = torch.tensor([127], dtype=torch.int32, device="cpu")
    page_table = torch.tensor([[0]], dtype=torch.int32, device="cpu")

    with pytest.raises(ValueError, match="8 or 16"):
        _msa_fwd(
            q,
            k,
            v,
            blocks,
            cu_q,
            kv_len,
            qo_offset,
            page_table,
            max_seqlen_q=1,
            max_seqlen_k=128,
            causal=True,
        )


def _online_same_dtype_p_reference(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    block_indexes: torch.Tensor,
    *,
    causal: bool,
    qo_offset: int,
    kv_len: int,
    softmax_scale: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_f = q.float().cpu()
    k_f = k_pages.float().cpu()
    v_f = v_pages.float().cpu()
    blocks = block_indexes.cpu()
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = k_pages.shape[2]
    head_ratio = num_heads // num_kv_heads
    out = torch.zeros((total_q, num_heads, head_dim), dtype=torch.float32, device="cpu")
    lse = torch.full(
        (total_q, num_heads), -torch.inf, dtype=torch.float32, device="cpu"
    )
    p_scale = 256.0

    for q_idx in range(total_q):
        for head in range(num_heads):
            head_kv = head // head_ratio
            row_max = -torch.inf
            row_sum = 0.0
            acc = torch.zeros(head_dim, dtype=torch.float32, device="cpu")
            for logical_block in blocks[q_idx, head_kv].tolist():
                if logical_block < 0 or logical_block * 128 >= kv_len:
                    continue
                k_block = k_f[logical_block, :, head_kv]
                v_block = v_f[logical_block, :, head_kv]
                scores = torch.mv(k_block, q_f[q_idx, head]) * softmax_scale
                kv_positions = logical_block * 128 + torch.arange(128, device="cpu")
                valid = kv_positions < kv_len
                if causal:
                    valid = valid & (kv_positions <= q_idx + qo_offset)
                scores = scores.masked_fill(~valid, -torch.inf)

                block_max = scores.max()
                row_max_tensor = torch.as_tensor(
                    row_max, dtype=torch.float32, device="cpu"
                )
                new_max = torch.maximum(row_max_tensor, block_max)
                if not torch.isfinite(new_max):
                    continue
                correction = (
                    0.0
                    if not torch.isfinite(row_max_tensor)
                    else torch.exp(row_max_tensor - new_max).item()
                )
                weights = torch.exp(scores - new_max) * p_scale
                weights = torch.where(valid, weights, torch.zeros_like(weights))
                row_sum = row_sum * correction + weights.sum().item()
                acc *= correction
                acc += weights.to(dtype).float() @ v_block
                row_max = new_max.item()

            if row_sum > 0.0:
                out[q_idx, head] = acc / row_sum
                lse[q_idx, head] = row_max + torch.log(
                    torch.tensor(row_sum / p_scale, dtype=torch.float32, device="cpu")
                )

    return out.to(dtype), lse


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("num_pages", [8, 20])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (16, 1), (32, 2)])
@pytest.mark.parametrize("dtype", [dtype for dtype, _ in _DTYPES])
@pytest.mark.skipif(
    not _HAS_MUSA,
    reason="MUSA is required for MSA forward",
)
def test_msa_fwd_uses_each_querys_own_topk(
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
    causal: bool,
    num_pages: int,
):
    device = torch.device("musa")
    total_q = 4
    kv_len = num_pages * 128
    softmax_scale = 128**-0.5

    torch.manual_seed(20260719)
    q = (torch.randn((total_q, num_q_heads, 128), device=device) * 0.25).to(dtype)
    k = (torch.randn((num_pages, 128, num_kv_heads, 128), device=device) * 0.25).to(
        dtype
    )
    v = (torch.randn((num_pages, 128, num_kv_heads, 128), device=device) * 0.25).to(
        dtype
    )
    block_rows = []
    for query in range(total_q):
        head_rows = []
        for head_kv in range(num_kv_heads):
            valid_count = min(num_pages, 16)
            valid = (
                torch.arange(valid_count, dtype=torch.int32, device="cpu")
                + query * 5
                + head_kv * 3
            ) % num_pages
            invalid = torch.full(
                (16 - valid_count,), -1, dtype=torch.int32, device="cpu"
            )
            head_rows.append(torch.cat((valid, invalid)))
        block_rows.append(torch.stack(head_rows))
    block_indexes_cpu = torch.stack(block_rows)
    block_indexes = block_indexes_cpu.to(device)
    cu_seqlens_q = torch.tensor([0, total_q], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)
    q_offset_value = kv_len - total_q if causal else 0
    qo_offset = torch.tensor([q_offset_value], dtype=torch.int32, device=device)
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(
        1, num_pages
    )

    out, lse = _msa_fwd(
        q,
        k,
        v,
        block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        max_seqlen_q=total_q,
        max_seqlen_k=kv_len,
        causal=causal,
        softmax_scale=softmax_scale,
    )
    torch.musa.synchronize()

    expected_out, expected_lse = _online_same_dtype_p_reference(
        q,
        k,
        v,
        block_indexes_cpu,
        causal=causal,
        qo_offset=q_offset_value,
        kv_len=kv_len,
        softmax_scale=softmax_scale,
        dtype=dtype,
    )
    actual = out.float().cpu()
    expected = expected_out.float()
    diff = actual - expected
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(-1), expected.reshape(-1), dim=0
    ).item()
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    rms = diff.square().mean().sqrt().item()
    min_cosine = 0.98 if dtype == _FP8 else 0.995
    max_abs_limit = (
        0.5 if dtype == _FP8 else (0.03 if dtype == torch.bfloat16 else 0.01)
    )
    assert out.dtype == dtype
    assert cosine > min_cosine, (
        f"cosine={cosine:.8f}, max_abs={max_abs:.8f}, "
        f"mean_abs={mean_abs:.8f}, rms={rms:.8f}"
    )
    assert max_abs < max_abs_limit, (
        f"cosine={cosine:.8f}, max_abs={max_abs:.8f}, "
        f"mean_abs={mean_abs:.8f}, rms={rms:.8f}"
    )

    lse_diff = lse.float().cpu() - expected_lse
    lse_limit = 0.1 if dtype == _FP8 else 0.02
    assert lse_diff.abs().max().item() < lse_limit


@pytest.mark.skipif(
    not _HAS_MUSA,
    reason="MUSA is required for MSA forward",
)
def test_msa_fwd_tp8_pair_union_handles_odd_varlen_and_disjoint_lists():
    device = torch.device("musa")
    dtype = torch.bfloat16
    q_lens = (3, 2)
    total_q = sum(q_lens)
    num_pages = 32
    kv_len = num_pages * 128
    softmax_scale = 128**-0.5

    torch.manual_seed(20260723)
    q = (torch.randn((total_q, 8, 128), device=device) * 0.25).to(dtype)
    k = (torch.randn((2 * num_pages, 128, 1, 128), device=device) * 0.25).to(dtype)
    v = (torch.randn((2 * num_pages, 128, 1, 128), device=device) * 0.25).to(dtype)

    block_indexes_cpu = torch.full(
        (total_q, 1, 16), -1, dtype=torch.int32, device="cpu"
    )
    # Batch 0: q0/q1 fully overlap, then an odd tail query.
    block_indexes_cpu[0, 0] = torch.arange(16, dtype=torch.int32, device="cpu")
    block_indexes_cpu[1, 0] = torch.arange(16, dtype=torch.int32, device="cpu")
    block_indexes_cpu[2, 0] = torch.arange(8, 24, dtype=torch.int32, device="cpu")
    # Batch 1: the paired queries have completely disjoint lists (union=32).
    block_indexes_cpu[3, 0] = torch.arange(16, dtype=torch.int32, device="cpu")
    block_indexes_cpu[4, 0] = torch.arange(16, 32, dtype=torch.int32, device="cpu")
    block_indexes = block_indexes_cpu.to(device)

    cu_seqlens_q = torch.tensor(
        [0, q_lens[0], total_q], dtype=torch.int32, device=device
    )
    seqused_k = torch.tensor([kv_len, kv_len], dtype=torch.int32, device=device)
    qo_offset = torch.zeros((2,), dtype=torch.int32, device=device)
    page_table = torch.arange(2 * num_pages, dtype=torch.int32, device=device).view(
        2, num_pages
    )

    out, lse = _msa_fwd(
        q,
        k,
        v,
        block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=kv_len,
        causal=False,
        softmax_scale=softmax_scale,
    )
    torch.musa.synchronize()

    expected_out_parts = []
    expected_lse_parts = []
    q_begin = 0
    for batch_idx, q_len in enumerate(q_lens):
        q_end = q_begin + q_len
        expected_out, expected_lse = _online_same_dtype_p_reference(
            q[q_begin:q_end],
            k[batch_idx * num_pages : (batch_idx + 1) * num_pages],
            v[batch_idx * num_pages : (batch_idx + 1) * num_pages],
            block_indexes_cpu[q_begin:q_end],
            causal=False,
            qo_offset=0,
            kv_len=kv_len,
            softmax_scale=softmax_scale,
            dtype=dtype,
        )
        expected_out_parts.append(expected_out)
        expected_lse_parts.append(expected_lse)
        q_begin = q_end

    expected_out = torch.cat(expected_out_parts)
    expected_lse = torch.cat(expected_lse_parts)
    actual = out.float().cpu()
    diff = actual - expected_out.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(-1), expected_out.float().reshape(-1), dim=0
    ).item()
    assert cosine > 0.995, (
        f"cosine={cosine:.8f}, max_abs={diff.abs().max().item():.8f}, "
        f"mean_abs={diff.abs().mean().item():.8f}"
    )
    assert diff.abs().max().item() < 0.03
    assert (lse.float().cpu() - expected_lse).abs().max().item() < 0.02


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (16, 1), (32, 2)])
@pytest.mark.parametrize("dtype", [dtype for dtype, _ in _DTYPES])
@pytest.mark.skipif(
    not _HAS_MUSA,
    reason="MUSA is required for MSA forward",
)
def test_msa_fwd_causal_partial_block_column_mapping(
    dtype: torch.dtype, num_q_heads: int, num_kv_heads: int
):
    device = torch.device("musa")
    q_len = kv_len = 128
    q = torch.zeros((q_len, num_q_heads, 128), dtype=dtype, device=device)
    k = torch.zeros((1, 128, num_kv_heads, 128), dtype=dtype, device=device)
    v = torch.zeros((1, 128, num_kv_heads, 128), dtype=torch.float32, device=device)
    v[0, :, :, 0] = torch.arange(128, dtype=torch.float32, device=device).view(128, 1)
    v = v.to(dtype)
    block_indexes = torch.full(
        (q_len, num_kv_heads, 16), -1, dtype=torch.int32, device=device
    )
    block_indexes[:, :, 0] = 0
    cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([kv_len], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([0], dtype=torch.int32, device=device)
    page_table = torch.tensor([0], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)

    out, lse = _msa_fwd(
        q,
        k,
        v,
        block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        kv_page_indptr=page_indptr,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        causal=True,
        softmax_scale=128**-0.5,
    )

    v_quantized = v[0, :, 0, 0].float().cpu()
    count = torch.arange(1, q_len + 1, dtype=torch.float32, device="cpu")
    expected = (v_quantized.cumsum(0) / count).to(dtype).float()
    torch.testing.assert_close(out[:, 0, 0].float().cpu(), expected)
    torch.testing.assert_close(
        lse[:, 0].float().cpu(), count.log(), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(out[:, -1, 0].float().cpu(), expected)
    torch.testing.assert_close(
        lse[:, -1].float().cpu(), count.log(), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("dtype", [dtype for dtype, _ in _DTYPES])
@pytest.mark.skipif(
    not _HAS_MUSA,
    reason="MUSA is required for MSA forward",
)
def test_msa_fwd_v_scale_is_applied_before_dtype_cast(dtype: torch.dtype):
    device = torch.device("musa")
    q = torch.zeros((1, 8, 128), dtype=dtype, device=device)
    k = torch.zeros((1, 128, 1, 128), dtype=dtype, device=device)
    v = torch.full((1, 128, 1, 128), 2.0, dtype=torch.float32, device=device).to(dtype)
    block_indexes = torch.full((1, 1, 16), -1, dtype=torch.int32, device=device)
    block_indexes[:, :, 0] = 0
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([1], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([0], dtype=torch.int32, device=device)
    page_table = torch.tensor([0], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)

    out, lse = _msa_fwd(
        q,
        k,
        v,
        block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        kv_page_indptr=page_indptr,
        max_seqlen_q=1,
        max_seqlen_k=1,
        causal=True,
        softmax_scale=128**-0.5,
        v_scale=1.5,
    )
    torch.testing.assert_close(
        out[0, 0, 0].float().cpu(),
        torch.tensor(3.0, dtype=torch.float32, device="cpu"),
        atol=0.2,
        rtol=0.05,
    )
    torch.testing.assert_close(
        out[0, -1, 0].float().cpu(),
        torch.tensor(3.0, dtype=torch.float32, device="cpu"),
        atol=0.2,
        rtol=0.05,
    )
    torch.testing.assert_close(
        lse[0].float().cpu(),
        torch.zeros(8, dtype=torch.float32, device="cpu"),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("dtype", [dtype for dtype, _ in _DTYPES])
@pytest.mark.skipif(
    not _HAS_MUSA,
    reason="MUSA is required for MSA forward",
)
def test_msa_fwd_k_scale_is_folded_into_softmax(dtype: torch.dtype):
    device = torch.device("musa")
    q = torch.zeros((1, 8, 128), dtype=torch.float32, device=device)
    q[:, :, 0] = 1.0
    q = q.to(dtype)
    k = torch.zeros((1, 128, 1, 128), dtype=torch.float32, device=device)
    k[0, 1, 0, 0] = 1.0
    k = k.to(dtype)
    v = torch.zeros((1, 128, 1, 128), dtype=torch.float32, device=device)
    v[0, 1, 0, 0] = 1.0
    v = v.to(dtype)
    block_indexes = torch.full((1, 1, 16), -1, dtype=torch.int32, device=device)
    block_indexes[:, :, 0] = 0
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    seqused_k = torch.tensor([2], dtype=torch.int32, device=device)
    qo_offset = torch.tensor([1], dtype=torch.int32, device=device)
    page_table = torch.tensor([0], dtype=torch.int32, device=device)
    page_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)

    out, lse = _msa_fwd(
        q,
        k,
        v,
        block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        kv_page_indptr=page_indptr,
        max_seqlen_q=1,
        max_seqlen_k=2,
        causal=True,
        softmax_scale=1.0,
        k_scale=2.0,
    )
    expected_out = torch.sigmoid(torch.tensor(2.0, dtype=torch.float32, device="cpu"))
    expected_lse = torch.logsumexp(
        torch.tensor([0.0, 2.0], dtype=torch.float32, device="cpu"), dim=0
    )
    torch.testing.assert_close(
        out[0, :, 0].float().cpu(),
        expected_out.expand(8),
        atol=0.03,
        rtol=0.03,
    )
    torch.testing.assert_close(
        lse[0].float().cpu(), expected_lse.expand(8), atol=0.02, rtol=0.02
    )
