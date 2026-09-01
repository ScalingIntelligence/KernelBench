# ruff: noqa
import inspect
from pathlib import Path
import random

import torch
import pytest
import tilelang.testing

from sparse_mla_test_utils import (
    FP8KVCacheLayout,
    _expected_flashmla_num_splits,
    _ref_sparse_mla_decode_model1,
    _ref_sparse_mla_prefill_v3_features,
    check_is_allclose,
    dequantize_k_cache,
    get_test_device,
    quantize_k_cache,
    ref_sparse_mla_fwd_interface,
)
# torch.random.manual_seed(42)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
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

    import flash_mla

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
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

    tl_out, lse = flash_mla.flash_mla_with_kvcache(
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
        tl_out.reshape(-1),
        ref_out.to(device).reshape(-1),
        abs_tol=2e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert is_out_correct


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_dsa_decode_topk_length():
    device = get_test_device()
    torch.manual_seed(20260428)
    random.seed(20260428)
    batch, sq, skv, heads, hkv, dqk, dv, topk = 3, 2, 1024, 64, 1, 576, 512, 128
    dtype = torch.bfloat16
    sm_scale = 0.0625
    pagesize = 64
    pagenum = (skv + pagesize - 1) // pagesize

    q = (torch.randn((batch, sq, heads, dqk), dtype=dtype, device=device) / 10).clamp_(
        -10, 10
    )
    kv = (
        torch.randn((pagenum, pagesize, hkv, dqk), dtype=dtype, device=device) / 10
    ).clamp_(-10, 10)
    indices = torch.randint(
        -16, skv + 16, (batch, sq, hkv, topk), dtype=torch.int32, device=device
    )
    topk_length = torch.tensor([0, 17, topk], dtype=torch.int32, device=device)
    attn_sink = torch.randn((heads,), dtype=torch.float32, device=device)
    attn_sink[::13] = float("inf")
    attn_sink[1::17] = float("-inf")

    kcache = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = dequantize_k_cache(kcache, FP8KVCacheLayout.V32_FP8Sparse).contiguous()

    import flash_mla

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=sq * heads,
        num_heads_k=hkv,
        num_heads_q=heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
        topk_length=topk_length,
    )
    tl_out, _ = flash_mla.flash_mla_with_kvcache(
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
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
    torch.musa.synchronize()

    masked_indices = indices.clone()
    pos = torch.arange(topk, device=device).view(1, 1, 1, topk)
    masked_indices = torch.where(
        pos < topk_length.view(batch, 1, 1, 1),
        masked_indices,
        masked_indices.new_full((), -1),
    )
    ref_out, ref_score = ref_sparse_mla_fwd_interface(
        q.view(batch * sq, heads, dqk),
        kv_dequant.view(pagenum * pagesize, hkv, -1),
        masked_indices.view(batch * sq, hkv, topk),
        sm_scale=sm_scale,
    )
    ref_lse = torch.logsumexp(ref_score[0].transpose(0, 1), dim=-1)
    ref_out = ref_out.float() * torch.sigmoid(
        ref_lse - attn_sink.view(1, heads)
    ).unsqueeze(-1)
    ref_out = torch.nan_to_num(ref_out, nan=0.0).to(dtype)
    assert check_is_allclose(
        "decode_topk_length_attn_sink_output",
        tl_out.reshape(-1),
        ref_out.to(device).reshape(-1),
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )


OFFICIAL_STYLE_V32_DECODE_CASES = [
    # Compact subset of FlashMLA sparse decode axes: S_q=1/S_q>1, odd page
    # sizes, topk > S_k, dynamic topk length, attn_sink, all-invalid rows, and
    # mixed zero-seqlen batches represented by precomputed sparse indices.
    ("page2_s1", 3, 1, 128, 64, 2, False, False, False, False),
    ("page61_s3", 2, 3, 183, 64, 61, True, False, False, False),
    ("page64_topk_gt_skv_sink", 3, 2, 153, 256, 64, True, True, False, False),
    ("page69_all_invalid", 2, 3, 207, 64, 69, True, True, True, False),
    ("page53_mixed_zero_seqlen", 4, 3, 159, 64, 53, False, True, False, True),
    (
        "page64_multi_batch_meta_dynamic",
        65,
        1,
        384,
        128,
        64,
        True,
        True,
        False,
        False,
    ),
]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "case",
    OFFICIAL_STYLE_V32_DECODE_CASES,
    ids=[case[0] for case in OFFICIAL_STYLE_V32_DECODE_CASES],
)
def test_v32_sparse_mla_decode_official_style(case):
    (
        tag,
        batch_size,
        seq_len_q,
        seq_len_kv,
        topk,
        page_size,
        have_topk_length,
        have_attn_sink,
        all_indices_invalid,
        have_zero_seqlen_k,
    ) = case
    device = get_test_device()
    torch.manual_seed(20260501 + page_size + seq_len_q)
    num_heads = 64
    d_qk = 576
    d_v = 512
    num_pages = (seq_len_kv + page_size - 1) // page_size

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn(
        (num_pages, page_size, 1, d_qk), dtype=torch.bfloat16, device=device
    )
    indices = torch.full(
        (batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device
    )
    if all_indices_invalid:
        indices.fill_(2147483647)
    else:
        valid_topk = min(topk, seq_len_kv)
        for batch_idx in range(batch_size):
            for q_idx in range(seq_len_q):
                indices[batch_idx, q_idx, 0, :valid_topk] = torch.randperm(
                    seq_len_kv, device=device
                )[:valid_topk]
        indices[:, :, :, 7::19] = -1
        indices[:, :, :, 11::23] = seq_len_kv + 5
        if have_zero_seqlen_k:
            indices[0].fill_(-1)

    topk_length = None
    if have_topk_length:
        if tag == "page64_multi_batch_meta_dynamic":
            topk_length = (
                torch.arange(1, batch_size + 1, dtype=torch.int32, device=device)
                .clamp_max(topk)
                .contiguous()
            )
        else:
            topk_length = torch.randint(
                0 if all_indices_invalid else 1,
                topk + 1,
                (batch_size,),
                dtype=torch.int32,
                device=device,
            )
        if all_indices_invalid:
            topk_length.zero_()
        elif have_zero_seqlen_k:
            topk_length[0] = 0

    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn((num_heads,), dtype=torch.float32, device=device)
        attn_sink[::17] = float("inf")
        attn_sink[1::19] = float("-inf")

    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.V32_FP8Sparse).view(
        -1, 1, d_qk
    )

    import flash_mla

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=seq_len_q * num_heads,
        num_heads_k=1,
        num_heads_q=num_heads,
        topk=topk,
        is_fp8_kvcache=True,
        q=q,
        topk_length=topk_length,
    )
    tl_out, tl_lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache.view(num_pages, page_size, 1, 656),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=d_qk**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
    if tag == "page64_multi_batch_meta_dynamic":
        assert tile_scheduler_metadata.tile_scheduler_metadata.shape[0] < batch_size
        assert (
            tile_scheduler_metadata.tile_scheduler_metadata[:, 2]
            > tile_scheduler_metadata.tile_scheduler_metadata[:, 0]
        ).any()
    torch.musa.synchronize()

    ref_topk_length = (
        topk_length.repeat_interleave(seq_len_q) if topk_length is not None else None
    )
    ref_out, _, ref_lse = _ref_sparse_mla_prefill_v3_features(
        q.view(batch_size * seq_len_q, num_heads, d_qk),
        kv_dequant,
        indices.view(batch_size * seq_len_q, 1, topk),
        sm_scale=d_qk**-0.5,
        topk_length=ref_topk_length,
        attn_sink=attn_sink,
        d_v=d_v,
    )
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, d_v)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)

    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_v32_sparse_mla_decode_large_cache_stride():
    device = get_test_device()
    torch.manual_seed(20260725)
    batch_size, seq_len_q, num_heads, d_qk, d_v, topk = 1, 1, 64, 576, 512, 64

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    kv = torch.randn((2, 1, 1, d_qk), dtype=torch.bfloat16, device=device)
    compact_cache = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse).contiguous()

    # Put the second logical token beyond the signed int32 address space.
    large_cache = torch.empty_strided(
        compact_cache.shape,
        (1 << 31, 656, 656, 1),
        dtype=compact_cache.dtype,
        device=device,
    )
    large_cache.copy_(compact_cache)
    indices = torch.ones(
        (batch_size, seq_len_q, 1, topk), dtype=torch.int32, device=device
    )

    import flash_mla

    sched_meta, _ = flash_mla.get_mla_metadata()

    def run_decode(k_cache):
        return flash_mla.flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=d_v,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            softmax_scale=d_qk**-0.5,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
        )

    compact_out, compact_lse = run_decode(compact_cache)
    large_out, large_lse = run_decode(large_cache)
    torch.musa.synchronize()

    torch.testing.assert_close(large_out, compact_out, rtol=0, atol=0)
    torch.testing.assert_close(large_lse, compact_lse, rtol=0, atol=0)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_v32_sparse_mla_decode_sched_meta_reuse_official_style():
    device = get_test_device()
    torch.manual_seed(20260507)
    batch_size, seq_len_q, seq_len_kv, topk, page_size = 2, 1, 128, 64, 64
    num_heads, d_qk, d_v = 64, 576, 512
    num_pages = (seq_len_kv + page_size - 1) // page_size
    kv = torch.randn(
        (num_pages, page_size, 1, d_qk), dtype=torch.bfloat16, device=device
    )
    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.V32_FP8Sparse).view(
        -1, 1, d_qk
    )

    def make_decode_inputs(seed):
        torch.manual_seed(seed)
        q = torch.randn(
            (batch_size, seq_len_q, num_heads, d_qk),
            dtype=torch.bfloat16,
            device=device,
        )
        indices = torch.full(
            (batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device
        )
        for batch_idx in range(batch_size):
            indices[batch_idx, 0, 0] = torch.randperm(seq_len_kv, device=device)[:topk]
        indices[:, :, :, 5::17] = -1
        indices[:, :, :, 11::19] = seq_len_kv + 7
        return q, indices

    import flash_mla

    sched_meta, _ = flash_mla.get_mla_metadata()
    q1, indices1 = make_decode_inputs(202605071)
    flash_mla.flash_mla_with_kvcache(
        q=q1,
        k_cache=k_cache.view(num_pages, page_size, 1, 656),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=d_qk**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices1,
    )
    first_metadata = sched_meta.tile_scheduler_metadata
    first_num_splits = sched_meta.num_splits
    assert first_metadata is not None
    assert first_num_splits is not None

    q2, indices2 = make_decode_inputs(202605072)
    tl_out, tl_lse = flash_mla.flash_mla_with_kvcache(
        q=q2,
        k_cache=k_cache.view(num_pages, page_size, 1, 656),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=d_qk**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices2,
    )
    assert sched_meta.tile_scheduler_metadata is first_metadata
    assert sched_meta.num_splits is first_num_splits
    torch.musa.synchronize()

    ref_out, _, ref_lse = _ref_sparse_mla_prefill_v3_features(
        q2.view(batch_size * seq_len_q, num_heads, d_qk),
        kv_dequant,
        indices2.view(batch_size * seq_len_q, 1, topk),
        sm_scale=d_qk**-0.5,
        d_v=d_v,
    )
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, d_v)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
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

    import flash_mla

    tl_out, _, _ = flash_mla.flash_mla_sparse_fwd(
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


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_dsa_prefill_v32_optional_params():
    device = get_test_device()
    sq, skv, heads, hkv, dqk, dv, topk = 37, 256, 64, 1, 576, 512, 128
    sm_scale = 0.1352337788608801
    q = (
        torch.randn((sq, heads, dqk), dtype=torch.bfloat16, device=device) / 10
    ).contiguous()
    kv = (
        torch.randn((skv, hkv, dqk), dtype=torch.bfloat16, device=device) / 10
    ).contiguous()
    indices = torch.randint(0, skv, (sq, hkv, topk), dtype=torch.int32, device=device)
    indices[::7, :, 9::13] = -1
    indices[5::11, :, 17::19] = skv + 3
    topk_length = torch.randint(1, topk + 1, (sq,), dtype=torch.int32, device=device)
    topk_length[::10] = 0
    attn_sink = torch.randn((heads,), dtype=torch.float32, device=device)
    attn_sink[::17] = float("inf")
    attn_sink[1::19] = float("-inf")

    import flash_mla

    tl_out, tl_max_logits, tl_lse = flash_mla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=dv,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    torch.musa.synchronize()

    ref_out, ref_max_logits, ref_lse = _ref_sparse_mla_prefill_v3_features(
        q, kv, indices, sm_scale, topk_length=topk_length, attn_sink=attn_sink, d_v=dv
    )
    valid_rows = torch.isfinite(ref_lse)
    assert check_is_allclose(
        "output",
        tl_out.reshape(-1),
        ref_out.to(device).reshape(-1),
        abs_tol=1e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert torch.allclose(tl_lse[valid_rows], ref_lse[valid_rows], atol=2e-3, rtol=2e-3)
    assert torch.isinf(tl_lse[~valid_rows]).all()
    assert torch.allclose(
        tl_max_logits[valid_rows], ref_max_logits[valid_rows], atol=2e-3, rtol=2e-3
    )
    assert torch.isneginf(tl_max_logits[~valid_rows]).all()


OFFICIAL_STYLE_PREFILL_CASES = [
    # Mirrors the official sparse prefill axes: d_qk, S_q shape, irregular S_k,
    # dynamic topk length, all-invalid rows, and attn_sink.
    ("v32_s1", 576, 1, 128, 64, 64, False, False, False),
    ("v32_dynamic_sink", 576, 62, 592, 128, 128, True, True, False),
    ("v32_all_invalid", 576, 213, 153, 256, 64, True, True, True),
    ("model1_s1", 512, 1, 128, 128, 64, False, False, False),
    ("model1_dynamic_sink", 512, 213, 153, 256, 128, True, True, False),
    ("model1_all_invalid", 512, 1, 128, 128, 64, True, True, True),
]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "case",
    OFFICIAL_STYLE_PREFILL_CASES,
    ids=[case[0] for case in OFFICIAL_STYLE_PREFILL_CASES],
)
def test_sparse_mla_prefill_official_style(case):
    (
        tag,
        d_qk,
        seq_len,
        seq_len_kv,
        topk,
        num_heads,
        have_topk_length,
        have_attn_sink,
        all_indices_invalid,
    ) = case
    del tag

    device = get_test_device()
    torch.manual_seed(20260429 + d_qk + seq_len)
    sm_scale = d_qk**-0.5
    q = torch.randn(
        (seq_len, num_heads, d_qk), dtype=torch.bfloat16, device=device
    ).contiguous()
    kv = torch.randn(
        (seq_len_kv, 1, d_qk), dtype=torch.bfloat16, device=device
    ).contiguous()

    indices = torch.full((seq_len, 1, topk), -1, dtype=torch.int32, device=device)
    if all_indices_invalid:
        indices.fill_(2147483647)
    elif d_qk == 512:
        for token_idx in range(seq_len):
            cur_len = min(topk, max(1, token_idx + 1))
            indices[token_idx, 0, :cur_len] = torch.randperm(
                max(1, token_idx + 1), device=device
            )[:cur_len]
    else:
        for token_idx in range(seq_len):
            cur_len = min(topk, seq_len_kv)
            indices[token_idx, 0, :cur_len] = torch.randperm(seq_len_kv, device=device)[
                :cur_len
            ]
        indices[::7, :, 5::11] = -1
        indices[5::13, :, 3::17] = seq_len_kv + 9

    topk_length = None
    if have_topk_length:
        topk_length = torch.randint(
            0, topk + 1, (seq_len,), dtype=torch.int32, device=device
        )
        if d_qk == 512 and not all_indices_invalid:
            topk_length.clamp_min_(1)
        else:
            topk_length[::9] = 0

    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn((num_heads,), dtype=torch.float32, device=device)
        attn_sink[::17] = float("inf")
        attn_sink[1::19] = float("-inf")

    import flash_mla

    tl_out, tl_max_logits, tl_lse = flash_mla.flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )
    torch.musa.synchronize()

    ref_out, ref_max_logits, ref_lse = _ref_sparse_mla_prefill_v3_features(
        q,
        kv,
        indices,
        sm_scale,
        topk_length=topk_length,
        attn_sink=attn_sink,
        d_v=512,
    )
    valid_rows = torch.isfinite(ref_lse)

    assert check_is_allclose(
        "official_prefill_output",
        tl_out.reshape(-1),
        ref_out.to(device).reshape(-1),
        abs_tol=4e-3,
        rel_tol=2.01 / 128,
        cos_diff_tol=5e-6,
    )
    assert torch.allclose(tl_lse[valid_rows], ref_lse[valid_rows], atol=2e-3, rtol=2e-3)
    assert torch.isinf(tl_lse[~valid_rows]).all()
    assert torch.allclose(
        tl_max_logits[valid_rows], ref_max_logits[valid_rows], atol=2e-3, rtol=2e-3
    )
    assert torch.isneginf(tl_max_logits[~valid_rows]).all()


MODEL1_PREFILL_CASES = [
    # (tag, S, SKV, topk, H, topk_len, sink, mostly_invalid, all_invalid, future_indices)
    ("basic_small", 1, 128, 128, 64, False, False, False, False, False),
    ("basic_oob", 213, 95, 128, 128, False, False, False, False, False),
    ("dynamic_len", 321, 512, 128, 128, True, False, False, False, False),
    ("attn_sink", 213, 153, 256, 64, True, True, False, False, False),
    ("corner_many_oob", 1024, 1024, 2048, 64, False, False, False, False, False),
    ("all_invalid", 321, 512, 128, 128, True, True, False, True, False),
]


MODEL1_DECODE_CASES = [
    # (tag, B, S, SKV, topk, SKV_EXTRA, extra_topk, H, topk_len, extra_topk_len, sink)
    ("basic", 128, 1, 8192, 2048, 0, 0, 128, False, False, False),
    ("tp8_local_h8_topk512_sink", 128, 1, 8192, 512, 0, 0, 8, True, False, True),
    ("basic_small", 4, 1, 512, 64, 0, 0, 64, False, False, False),
    ("batch_topk_len_s3", 4, 3, 512, 64, 0, 0, 64, True, False, False),
    ("extra", 74, 1, 1024, 576, 1024, 576, 128, False, False, False),
    ("dynamic_len", 321, 1, 2046, 2048, 2046, 2048, 128, True, True, False),
    ("attn_sink", 32, 1, 1024, 576, 1024, 576, 128, True, True, True),
    ("all_invalid", 32, 1, 512, 64, 512, 64, 64, True, True, True),
]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "case",
    [
        ("no_extra", 4, 64, None, 0, None),
        ("extra_static", 4, 64, None, 64, None),
        (
            "dynamic_length",
            4,
            128,
            [0, 1, 64, 127],
            64,
            [0, 7, 63, 64],
        ),
        ("all_invalid", 4, 64, [0, 0, 0, 0], 64, [0, 0, 0, 0]),
    ],
    ids=lambda case: case[0],
)
def test_model1_metadata_dynamic_topk_lengths(case):
    tag, batch_size, topk, topk_len_values, extra_topk, extra_len_values = case
    del tag
    device = get_test_device()
    num_heads = 64
    q = torch.empty(
        (batch_size, 1, num_heads, 512), dtype=torch.bfloat16, device=device
    )
    topk_length = (
        torch.tensor(topk_len_values, dtype=torch.int32, device=device)
        if topk_len_values is not None
        else None
    )
    extra_topk_length = (
        torch.tensor(extra_len_values, dtype=torch.int32, device=device)
        if extra_len_values is not None
        else None
    )

    from mate.flashmla import get_mla_metadata

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=num_heads,
        num_heads_k=1,
        num_heads_q=num_heads,
        is_fp8_kvcache=True,
        topk=topk + extra_topk,
        q=q,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    expected = _expected_flashmla_num_splits(
        topk=topk,
        batch_size=batch_size,
        num_mp_parts=int(tile_scheduler_metadata.shape[0]),
        topk_length=topk_length,
        extra_topk=extra_topk,
        extra_topk_length=extra_topk_length,
    )
    assert num_splits.cpu().tolist() == expected


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
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

    import flash_mla

    tl_out, _, tl_lse = flash_mla.flash_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        d_v=512,
        attn_sink=attn_sink,
        topk_length=topk_length,
    )

    ref_out, _, ref_lse = _ref_sparse_mla_prefill_v3_features(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
        d_v=512,
        topk_length=topk_length,
    )
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
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
            (num_extra_pages, page_size, 1, 512),
            dtype=torch.bfloat16,
            device=device,
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

    q_flat = q.view(total_q, num_heads, 512)
    indices_flat = indices.view(total_q, 1, topk)
    extra_indices_flat = (
        extra_indices.view(total_q, 1, extra_topk)
        if extra_indices is not None
        else None
    )

    import flash_mla

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=seq_len_q * num_heads // 1,
        num_heads_k=1,
        num_heads_q=num_heads,
        topk=topk,
        extra_topk=extra_topk if extra_topk > 0 else None,
        is_fp8_kvcache=True,
        q=q,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )

    tl_out, tl_lse = flash_mla.flash_mla_with_kvcache(
        q=q.view(batch_size, seq_len_q, num_heads, 512),
        k_cache=k_cache,
        indices=indices.view(batch_size, seq_len_q, 1, topk),
        block_table=None,
        head_dim_v=512,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices.view(
            batch_size, seq_len_q, 1, extra_topk
        )
        if extra_indices is not None
        else None,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        softmax_scale=0.1352337788608801,
        attn_sink=attn_sink,
        cache_seqlens=None,
        tile_scheduler_metadata=tile_scheduler_metadata,
        is_fp8_kvcache=True,
        num_splits=num_splits,
    )
    ref_out, ref_lse = _ref_sparse_mla_decode_model1(
        q_flat,
        kv_dequant,
        indices_flat,
        extra_kv=extra_kv_dequant,
        extra_indices=extra_indices_flat,
        topk_length=topk_length.repeat_interleave(seq_len_q)
        if topk_length is not None
        else None,
        extra_topk_length=extra_topk_length,
        sm_scale=0.1352337788608801,
        attn_sink=attn_sink,
    )
    tl_out = tl_out.view(batch_size, seq_len_q, num_heads, 512)
    tl_lse = tl_lse.view(batch_size, num_heads, seq_len_q)
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, 512)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


OFFICIAL_STYLE_MODEL1_DECODE_CASES = [
    # Mirrors the official tests/lib.py decode axes that matter for the wrapper:
    # S=1/S>1, irregular page sizes, extra KV, dynamic lengths, attn_sink, and
    # mixed zero-seqlen batches represented by precomputed sparse indices.
    ("page2_s1", 3, 1, 128, 64, 2, 0, 0, 2, False, False, False, False, False),
    ("page61_s3", 2, 3, 183, 64, 61, 0, 0, 61, True, False, False, False, False),
    (
        "page64_extra_sink",
        3,
        1,
        256,
        128,
        64,
        256,
        64,
        64,
        True,
        True,
        True,
        False,
        False,
    ),
    (
        "page69_all_invalid",
        2,
        2,
        207,
        64,
        69,
        207,
        64,
        69,
        True,
        True,
        True,
        True,
        False,
    ),
    (
        "page53_mixed_zero_seqlen",
        4,
        3,
        159,
        64,
        53,
        0,
        0,
        53,
        False,
        False,
        True,
        False,
        True,
    ),
]


MODEL1_DECODE_SCHEDULED_CASES = [
    ("no_extra_split", 512, 128, 0, 0, False, False, False),
    ("extra_dynamic_sink", 512, 128, 256, 64, True, True, False),
    ("all_invalid", 512, 128, 256, 64, True, True, True),
]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "case",
    MODEL1_DECODE_SCHEDULED_CASES,
    ids=[case[0] for case in MODEL1_DECODE_SCHEDULED_CASES],
)
def test_model1_sparse_mla_decode_scheduled(case):
    (
        tag,
        seq_len_kv,
        topk,
        seq_len_kv_extra,
        extra_topk,
        have_lengths,
        have_attn_sink,
        all_indices_invalid,
    ) = case

    torch.random.manual_seed(321)
    device = get_test_device()
    batch_size = 1
    seq_len_q = 1
    num_heads = 64
    page_size = 64
    total_q = batch_size * seq_len_q
    sm_scale = 0.1352337788608801

    q = torch.randn(
        (batch_size, seq_len_q, num_heads, 512), dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(
        ((seq_len_kv + page_size - 1) // page_size, page_size, 1, 512),
        dtype=torch.bfloat16,
        device=device,
    )
    indices = torch.full(
        (batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device
    )
    if not all_indices_invalid:
        indices[0, 0, 0] = torch.randperm(seq_len_kv, device=device)[:topk]
    else:
        indices.fill_(2147483647)

    topk_length = None
    if have_lengths:
        topk_length = torch.tensor(
            [0 if all_indices_invalid else topk - 32],
            dtype=torch.int32,
            device=device,
        )

    extra_k_cache = None
    extra_kv_dequant = None
    extra_indices = None
    extra_topk_length = None
    if extra_topk:
        extra_kv = torch.randn(
            ((seq_len_kv_extra + page_size - 1) // page_size, page_size, 1, 512),
            dtype=torch.bfloat16,
            device=device,
        )
        extra_indices = torch.full(
            (batch_size, seq_len_q, 1, extra_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        if not all_indices_invalid:
            extra_indices[0, 0, 0] = torch.randperm(seq_len_kv_extra, device=device)[
                :extra_topk
            ]
        else:
            extra_indices.fill_(2147483647)
        if have_lengths:
            extra_topk_length = torch.tensor(
                [0 if all_indices_invalid else extra_topk // 2],
                dtype=torch.int32,
                device=device,
            )
        extra_k_cache = quantize_k_cache(extra_kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
        extra_kv_dequant = dequantize_k_cache(
            extra_k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse
        ).view(-1, 1, 512)

    attn_sink = None
    if have_attn_sink:
        attn_sink = torch.randn((num_heads,), dtype=torch.float32, device=device)
        attn_sink[::5] = float("inf")
        attn_sink[1::7] = float("-inf")

    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse).view(
        -1, 1, 512
    )

    import flash_mla

    sched_meta, _ = flash_mla.get_mla_metadata()

    tl_out, tl_lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        indices=indices,
        block_table=None,
        head_dim_v=512,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        softmax_scale=sm_scale,
        attn_sink=attn_sink,
        cache_seqlens=None,
        tile_scheduler_metadata=sched_meta,
        is_fp8_kvcache=True,
        num_splits=None,
    )
    assert sched_meta.num_splits is not None
    if tag != "all_invalid":
        assert int(torch.diff(sched_meta.num_splits).max().item()) > 1

    q_flat = q.view(total_q, num_heads, 512)
    indices_flat = indices.view(total_q, 1, topk)
    extra_indices_flat = (
        extra_indices.view(total_q, 1, extra_topk)
        if extra_indices is not None
        else None
    )
    ref_out, ref_lse = _ref_sparse_mla_decode_model1(
        q_flat,
        kv_dequant,
        indices_flat,
        extra_kv=extra_kv_dequant,
        extra_indices=extra_indices_flat,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        sm_scale=sm_scale,
        attn_sink=attn_sink,
    )

    tl_out = tl_out.view(batch_size, seq_len_q, num_heads, 512)
    tl_lse = tl_lse.view(batch_size, num_heads, seq_len_q)
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, 512)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)

    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "case",
    OFFICIAL_STYLE_MODEL1_DECODE_CASES,
    ids=[case[0] for case in OFFICIAL_STYLE_MODEL1_DECODE_CASES],
)
def test_model1_sparse_mla_decode_official_style(case):
    (
        tag,
        batch_size,
        seq_len_q,
        seq_len_kv,
        topk,
        page_size,
        seq_len_kv_extra,
        extra_topk,
        extra_page_size,
        have_topk_length,
        have_extra_topk_length,
        have_attn_sink,
        all_indices_invalid,
        have_zero_seqlen_k,
    ) = case
    del tag

    torch.random.manual_seed(123)
    device = get_test_device()
    num_heads = 64
    total_q = batch_size * seq_len_q
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
                indices[batch_idx, q_idx, 0, : cur_indices.numel()] = cur_indices
        if have_zero_seqlen_k:
            indices[0].fill_(-1)
    else:
        indices.fill_(2147483647)

    topk_length = None
    if have_topk_length:
        high = topk + 1
        topk_length = torch.randint(
            0 if all_indices_invalid else 1,
            high,
            (batch_size,),
            dtype=torch.int32,
            device=device,
        )
        if all_indices_invalid:
            topk_length.zero_()
        elif have_zero_seqlen_k:
            topk_length[0] = 0

    extra_k_cache = None
    extra_kv_dequant = None
    extra_indices = None
    extra_topk_length = None
    if extra_topk > 0:
        num_extra_pages = (seq_len_kv_extra + extra_page_size - 1) // extra_page_size
        extra_kv = torch.randn(
            (num_extra_pages, extra_page_size, 1, 512),
            dtype=torch.bfloat16,
            device=device,
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
                    extra_indices[batch_idx, q_idx, 0, : cur_indices.numel()] = (
                        cur_indices
                    )
            if have_zero_seqlen_k:
                extra_indices[0].fill_(-1)
        else:
            extra_indices.fill_(2147483647)
        if have_extra_topk_length:
            extra_topk_length = torch.randint(
                0 if all_indices_invalid else 1,
                extra_topk + 1,
                (batch_size,),
                dtype=torch.int32,
                device=device,
            )
            if all_indices_invalid:
                extra_topk_length.zero_()
            elif have_zero_seqlen_k:
                extra_topk_length[0] = 0
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

    import flash_mla

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=seq_len_q * num_heads,
        num_heads_k=1,
        num_heads_q=num_heads,
        topk=topk,
        extra_topk=extra_topk if extra_topk > 0 else None,
        is_fp8_kvcache=True,
        q=q,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
    )
    tl_out, tl_lse = flash_mla.flash_mla_with_kvcache(
        q=q,
        k_cache=k_cache,
        indices=indices,
        block_table=None,
        head_dim_v=512,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        softmax_scale=0.1352337788608801,
        attn_sink=attn_sink,
        cache_seqlens=None,
        tile_scheduler_metadata=tile_scheduler_metadata,
        is_fp8_kvcache=True,
        num_splits=num_splits,
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


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_model1_sparse_mla_decode_sched_meta_reuse_official_style():
    device = get_test_device()
    torch.manual_seed(20260508)
    batch_size, seq_len_q = 2, 1
    seq_len_kv, topk, page_size = 256, 128, 64
    seq_len_kv_extra, extra_topk, extra_page_size = 256, 64, 64
    num_heads, d_qk, d_v = 64, 512, 512
    num_pages = (seq_len_kv + page_size - 1) // page_size
    num_extra_pages = (seq_len_kv_extra + extra_page_size - 1) // extra_page_size

    kv = torch.randn(
        (num_pages, page_size, 1, d_qk), dtype=torch.bfloat16, device=device
    )
    extra_kv = torch.randn(
        (num_extra_pages, extra_page_size, 1, d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    k_cache = quantize_k_cache(kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
    extra_k_cache = quantize_k_cache(extra_kv, FP8KVCacheLayout.MODEL1_FP8Sparse)
    kv_dequant = dequantize_k_cache(k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse).view(
        -1, 1, d_qk
    )
    extra_kv_dequant = dequantize_k_cache(
        extra_k_cache, FP8KVCacheLayout.MODEL1_FP8Sparse
    ).view(-1, 1, d_qk)

    def make_decode_inputs(seed):
        torch.manual_seed(seed)
        q = torch.randn(
            (batch_size, seq_len_q, num_heads, d_qk),
            dtype=torch.bfloat16,
            device=device,
        )
        indices = torch.full(
            (batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device
        )
        extra_indices = torch.full(
            (batch_size, seq_len_q, 1, extra_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for batch_idx in range(batch_size):
            indices[batch_idx, 0, 0] = torch.randperm(seq_len_kv, device=device)[:topk]
            extra_indices[batch_idx, 0, 0] = torch.randperm(
                seq_len_kv_extra, device=device
            )[:extra_topk]
        indices[:, :, :, 9::23] = -1
        indices[:, :, :, 13::29] = seq_len_kv + 3
        extra_indices[:, :, :, 5::17] = -1
        extra_indices[:, :, :, 11::19] = seq_len_kv_extra + 7
        return q, indices, extra_indices

    import flash_mla

    sched_meta, _ = flash_mla.get_mla_metadata()
    q1, indices1, extra_indices1 = make_decode_inputs(202605081)
    flash_mla.flash_mla_with_kvcache(
        q=q1,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=d_qk**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices1,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices1,
    )
    first_metadata = sched_meta.tile_scheduler_metadata
    first_num_splits = sched_meta.num_splits
    assert first_metadata is not None
    assert first_num_splits is not None

    q2, indices2, extra_indices2 = make_decode_inputs(202605082)
    tl_out, tl_lse = flash_mla.flash_mla_with_kvcache(
        q=q2,
        k_cache=k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=d_v,
        tile_scheduler_metadata=sched_meta,
        num_splits=None,
        softmax_scale=d_qk**-0.5,
        causal=False,
        is_fp8_kvcache=True,
        indices=indices2,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices2,
    )
    assert sched_meta.tile_scheduler_metadata is first_metadata
    assert sched_meta.num_splits is first_num_splits
    torch.musa.synchronize()

    total_q = batch_size * seq_len_q
    ref_out, ref_lse = _ref_sparse_mla_decode_model1(
        q2.view(total_q, num_heads, d_qk),
        kv_dequant,
        indices2.view(total_q, 1, topk),
        extra_kv=extra_kv_dequant,
        extra_indices=extra_indices2.view(total_q, 1, extra_topk),
        sm_scale=d_qk**-0.5,
    )
    ref_out = ref_out.view(batch_size, seq_len_q, num_heads, d_v)
    ref_lse = ref_lse.view(batch_size, seq_len_q, num_heads).transpose(1, 2)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(tl_lse, ref_lse.to(device), rtol=1e-2, atol=1e-2)
