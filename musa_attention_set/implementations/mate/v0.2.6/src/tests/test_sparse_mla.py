from __future__ import annotations

import pytest
import torch

import mate
from mate.sparse_mla_interface import (
    get_batch_decode_metadata_mla,
    mla_rope_quantize_fp8,
    sparse_mla_fp8_decode,
)
from mate.testing import supported_musa_compute_capability
from mate.testing.sparse_mla import (
    Fp8DecodeCase,
    RopeCase,
    SparseMlaCase,
    assert_sparse_mla_close,
    make_decode_data,
    make_fp8_decode_data,
    make_fp8_decode_stress_data,
    make_prefill_data,
    make_rope_data,
)


PREFILL_CASES = (
    SparseMlaCase("v32_regular", "v32", 1, 128, 64, 128),
    SparseMlaCase(
        "v32_features",
        "v32",
        62,
        95,
        128,
        128,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
    ),
    SparseMlaCase("v4_regular", "v4", 1, 256, 64, 256),
    SparseMlaCase(
        "v4_features",
        "v4",
        62,
        153,
        128,
        256,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
    ),
    SparseMlaCase("v32_large", "v32", 213, 1592, 128, 384, full=True),
    SparseMlaCase("v4_large", "v4", 213, 1840, 128, 256, full=True),
)


ROPE_CASES = (
    RopeCase("neox_fp32", 5, 3, True, torch.float32),
    RopeCase("interleaved_bf16", 62, 64, False, torch.bfloat16),
    RopeCase("neox_bf16", 19, 16, True, torch.bfloat16),
    RopeCase("large", 213, 128, False, torch.float32, full=True),
)


DECODE_CASES = (
    SparseMlaCase("v32_page2", "v32", 1, 512, 64, 64, batch=4, page_size=2),
    SparseMlaCase(
        "v32_irregular",
        "v32",
        3,
        650,
        128,
        576,
        batch=4,
        page_size=61,
        invalid_rows=True,
    ),
    SparseMlaCase("v4_page69", "v4", 1, 512, 64, 64, batch=4, page_size=69),
    SparseMlaCase(
        "v4_features",
        "v4",
        3,
        650,
        128,
        128,
        batch=4,
        page_size=61,
        topk_length=True,
        attn_sink=True,
        invalid_rows=True,
        extra_kv_len=512,
        extra_topk=64,
    ),
    SparseMlaCase(
        "v32_large",
        "v32",
        2,
        32768,
        128,
        2048,
        batch=74,
        full=True,
    ),
    SparseMlaCase(
        "v4_large",
        "v4",
        3,
        2046,
        128,
        2048,
        batch=74,
        full=True,
    ),
)


FP8_DECODE_CASES = (
    Fp8DecodeCase("q3_page61_4d", 2, 3, 183, 256, 61, 4),
    Fp8DecodeCase("mixed_invalid_page53_3d", 4, 2, 159, 64, 53, 3, True),
    Fp8DecodeCase("topk2112", 2, 1, 2176, 2112, 64, 4, full=True),
)


def _skip_large_case(case, pytestconfig: pytest.Config) -> None:
    if case.full and not pytestconfig.getoption("sparse_mla_full"):
        pytest.skip("enable with --sparse-mla-full")


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", PREFILL_CASES, ids=lambda case: case.name)
def test_sparse_mla_prefill(case: SparseMlaCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_prefill_data(case)

    out, max_logits, lse = mate.flashmla.flash_mla_sparse_fwd(
        q=data.q,
        kv=data.kv,
        indices=data.indices,
        sm_scale=case.head_dim**-0.5,
        d_v=512,
        attn_sink=data.attn_sink,
        topk_length=data.topk_length,
    )

    assert_sparse_mla_close(
        out,
        data.ref_out,
        lse,
        data.ref_lse,
        max_logits=max_logits,
        ref_max_logits=data.ref_max_logits,
    )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", ROPE_CASES, ids=lambda case: case.name)
def test_mla_rope_quantize_fp8(case: RopeCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_rope_data(case)
    q_merged = torch.empty(
        case.nnz, case.heads, 576, dtype=torch.float8_e4m3fn, device="musa"
    )
    k_merged = torch.empty(case.nnz, 576, dtype=torch.float8_e4m3fn, device="musa")

    outputs = mla_rope_quantize_fp8(
        data.q_rope,
        data.k_rope,
        data.q_nope,
        data.k_nope,
        data.cos_sin_cache,
        data.pos_ids,
        is_neox=case.is_neox,
        quant_scale_q=0.625,
        quant_scale_kv=1.75,
        q_rope_out=q_merged[..., 512:],
        k_rope_out=k_merged[..., 512:],
        q_nope_out=q_merged[..., :512],
        k_nope_out=k_merged[..., :512],
    )

    for actual, expected in zip(outputs, data.ref_outputs):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.01
        )


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", DECODE_CASES, ids=lambda case: case.name)
def test_sparse_mla_decode(case: SparseMlaCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_decode_data(case)
    metadata, num_splits = mate.flashmla.get_mla_metadata(
        cache_seqlens=None,
        num_q_tokens_per_head_k=case.q_len * case.heads,
        num_heads_k=1,
        num_heads_q=case.heads,
        is_fp8_kvcache=True,
        topk=case.topk,
        extra_topk=case.extra_topk or None,
        q=data.q,
        bs=case.batch,
        topk_length=data.topk_length,
        extra_topk_length=data.extra_topk_length,
    )

    out, lse = mate.flashmla.flash_mla_with_kvcache(
        q=data.q,
        k_cache=data.k_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=metadata,
        num_splits=num_splits,
        softmax_scale=case.head_dim**-0.5,
        is_fp8_kvcache=True,
        indices=data.indices,
        attn_sink=data.attn_sink,
        extra_k_cache=data.extra_k_cache,
        extra_indices_in_kvcache=data.extra_indices,
        topk_length=data.topk_length,
        extra_topk_length=data.extra_topk_length,
    )

    assert_sparse_mla_close(out, data.ref_out, lse, data.ref_lse)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("case", FP8_DECODE_CASES, ids=lambda case: case.name)
def test_sparse_mla_fp8_decode(case: Fp8DecodeCase, pytestconfig: pytest.Config):
    _skip_large_case(case, pytestconfig)
    data = make_fp8_decode_data(case)
    metadata = get_batch_decode_metadata_mla(data.query, data.seq_lens, case.topk)
    lse_buffer = torch.empty_like(data.ref_lse)

    out, lse = sparse_mla_fp8_decode(
        query=data.query,
        kv_cache=data.kv_cache,
        workspace_buffer=object(),
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        block_tables=data.indices,
        seq_lens=data.seq_lens,
        max_seq_len=case.kv_len,
        sparse_mla_top_k=case.topk,
        bmm1_scale=0.0625,
        lse=lse_buffer,
        return_lse=True,
        metadata=metadata,
    )

    assert_sparse_mla_close(out, data.ref_out, lse, data.ref_lse)


@supported_musa_compute_capability([31])
def test_sparse_mla_fp8_decode_stress(pytestconfig: pytest.Config):
    iterations = pytestconfig.getoption("sparse_mla_stress_iters")
    if iterations == 0:
        pytest.skip("enable with --sparse-mla-stress-iters=N")
    if iterations < 10000:
        raise pytest.UsageError("--sparse-mla-stress-iters must be 0 or >= 10000")

    data = make_fp8_decode_stress_data()
    out_buffer = torch.empty_like(data.ref_out)
    lse_buffer = torch.empty_like(data.ref_lse)
    metadata = get_batch_decode_metadata_mla(
        data.query, data.seq_lens, data.indices.shape[-1]
    )
    decode_kwargs = {
        "query": data.query,
        "kv_cache": data.kv_cache,
        "workspace_buffer": object(),
        "qk_nope_head_dim": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "block_tables": data.indices,
        "seq_lens": data.seq_lens,
        "max_seq_len": data.indices.shape[-1],
        "sparse_mla_top_k": data.indices.shape[-1],
        "out": out_buffer,
        "bmm1_scale": 0.0625,
        "lse": lse_buffer,
        "return_lse": True,
        "metadata": metadata,
    }

    baseline_out, baseline_lse = sparse_mla_fp8_decode(**decode_kwargs)
    torch.musa.synchronize()
    assert_sparse_mla_close(baseline_out, data.ref_out, baseline_lse, data.ref_lse)
    baseline_out = baseline_out.clone()
    baseline_lse = baseline_lse.clone()
    progress_interval = max(1000, iterations // 10)

    for iteration in range(1, iterations + 1):
        try:
            current_out, current_lse = sparse_mla_fp8_decode(**decode_kwargs)
            torch.musa.synchronize()
            if not torch.equal(current_out, baseline_out) or not torch.equal(
                current_lse, baseline_lse
            ):
                raise AssertionError("output or LSE changed")
        except Exception as exc:
            raise RuntimeError(
                "FP8 sparse decode persistent-row stress failed at "
                f"iteration {iteration}/{iterations}"
            ) from exc
        if iteration % progress_interval == 0:
            print(
                f"sparse MLA stress: {iteration}/{iterations} iterations",
                flush=True,
            )
