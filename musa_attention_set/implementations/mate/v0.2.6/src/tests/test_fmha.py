# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
from dataclasses import replace
from itertools import product
from typing import Any, Sequence

import pytest
import torch
import torch_musa  # noqa: F401

from mate import flash_attn_combine  # noqa: F401
from mate.execution_context import is_dry_run_enabled
from mate.testing.flash_attn import attention_ref
from mate.testing.operator import OpMode
from mate.testing.operator.fmha import (
    FmhaApi,
    FmhaInputMode,
    FmhaLayout,
    FmhaOperator,
    FmhaReference,
    FmhaVerifyMode,
    FmhaWorkload,
    UnsupportedFmhaWorkload,
)
from mate.testing import supported_musa_compute_capability


HEADS = ((40, 8), (96, 8), (32, 8), (64, 4), (8, 1))
HEAD_DIMS = ((128, 128), (256, 256))
RAGGED_SEQUENCES = (
    ((32, 512), (192, 192), (128, 256)),
    ((32, 512, 32, -1), (192, 192, 64, 128), (128, 256, 64, 128)),
    tuple((index, index + 1234) for index in range(11)),
    ((55, 666), (256, 463), (111, 1328)),
    ((111, 1328), (55, 666), (222, 463)),
)
PAGED_SEQUENCES = (
    ((32, 512), (192, 192), (128, 256)),
    tuple((index, index + 1234) for index in range(11)),
    ((55, 666), (222, 463), (111, 1328)),
    ((111, 1328), (55, 666), (222, 463)),
)
ADVANCED_RAGGED_SEQUENCES = (
    ((111, 1328), (55, 666), (222, 463)),
    ((111, 1328, 66, 888), (55, 666, 55, -1), (222, 463, -1, 256)),
)


def _sequence_options(sequence: Sequence[Sequence[int]]) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": len(sequence),
        "seqlen_q": max(item[0] for item in sequence),
        "seqlen_k": max(item[1] for item in sequence),
        "q_lengths": tuple(item[0] for item in sequence),
        "kv_lengths": tuple(item[1] for item in sequence),
    }
    if len(sequence[0]) == 4:
        options["q_used_lengths"] = tuple(item[2] for item in sequence)
        options["kv_used_lengths"] = tuple(item[3] for item in sequence)
    return options


def _mask_options(mask: str | tuple[int, int] | None) -> dict[str, Any]:
    if mask == "causal":
        return {"causal": True}
    if isinstance(mask, tuple):
        return {"window_size": mask}
    return {}


METADATA_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.KVCACHE,
        input_mode=FmhaInputMode.METADATA,
        batch_size=1000,
        seqlen_q=1,
        seqlen_k=16,
        q_lengths=(1,) * 1000,
        kv_lengths=(16,) * 1000,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        q_layout=FmhaLayout.RAGGED,
        kv_layout=FmhaLayout.PAGED,
        page_size=page_size,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        expect_single_split=True,
        **_mask_options(mask),
    )
    for (num_heads_q, num_heads_kv), (
        head_dim_qk,
        head_dim_v,
    ), page_size, mask, pack_gqa, num_splits in product(
        ((40, 8), (32, 8), (64, 4), (8, 1)),
        HEAD_DIMS,
        (1, 16, 64),
        (None, "causal"),
        (True, False),
        (0, 5),
    )
]

STRIDE_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.VARLEN,
        input_mode=FmhaInputMode.NONCONTIGUOUS,
        batch_size=len(lengths),
        seqlen_q=max(lengths),
        seqlen_k=max(lengths),
        q_lengths=lengths,
        kv_lengths=lengths,
        num_heads_q=24,
        num_heads_kv=24,
        head_dim_qk=64,
        head_dim_v=64,
        dtype=torch.float16,
        backend="mubin",
        return_softmax_lse=False,
        atol=1e-3,
        rtol=1e-3,
    )
    for lengths in ((128,), (64, 64))
] + [
    FmhaWorkload(
        api=FmhaApi.VARLEN,
        input_mode=FmhaInputMode.LARGE_STRIDE,
        batch_size=1,
        seqlen_q=128,
        seqlen_k=128,
        num_heads_q=2,
        num_heads_kv=2,
        head_dim_qk=64,
        head_dim_v=64,
        dtype=torch.float16,
        backend="mubin",
        return_softmax_lse=False,
        atol=1e-3,
        rtol=1e-3,
    )
]

DENSE_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.VARLEN,
        batch_size=1,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_q,
        head_dim_qk=128,
        head_dim_v=128,
        attention_chunk=attention_chunk,
        num_splits=num_splits,
        pack_gqa=False,
        backend=backend,
    )
    for (seqlen_q, seqlen_k), (
        num_heads_q,
        _,
    ), attention_chunk, num_splits, backend in product(
        ((333, 444), (888, 1328), (222, 463)),
        HEADS,
        (0, 65),
        (-1, 0, 5),
        ("auto", "mutlass"),
    )
]

FP8_SOFTCAP_ALL_MASKED_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.VARLEN,
        verify_mode=FmhaVerifyMode.ADVANCED,
        batch_size=9,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        dtype=torch.float8_e4m3fn,
        seed=0,
        causal=causal,
        window_size=(seqlen_k // 2, 0),
        attention_chunk=seqlen_k,
        softcap=15.0,
        num_splits=1,
        pack_gqa=False,
        return_softmax_lse=False,
        compare_lse=False,
    )
    for seqlen_q, seqlen_k, num_heads_q, num_heads_kv, head_dim, causal in (
        (1024, 1023, 6, 6, 128, False),
        (1024, 1023, 6, 2, 256, False),
        (1024, 1024, 6, 2, 128, True),
    )
]

RAGGED_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.VARLEN,
        q_layout=FmhaLayout.RAGGED,
        kv_layout=FmhaLayout.RAGGED,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        cp_world_size=cp_world_size,
        backend=backend,
        compare_lse=False,
        **_sequence_options(sequence),
        **_mask_options(mask),
    )
    for sequence, (num_heads_q, num_heads_kv), (
        head_dim_qk,
        head_dim_v,
    ), backend, mask, pack_gqa, num_splits, cp_world_size in product(
        RAGGED_SEQUENCES,
        HEADS,
        HEAD_DIMS,
        ("auto", "mutlass"),
        (None, "causal"),
        (False, True),
        (-1, 0, 5),
        (1, 2, 4),
    )
]

RAGGED_ADVANCED_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.VARLEN,
        q_layout=FmhaLayout.RAGGED,
        kv_layout=FmhaLayout.RAGGED,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        softcap=softcap,
        compare_lse=False,
        **_sequence_options(sequence),
        **_mask_options(mask),
    )
    for sequence, (num_heads_q, num_heads_kv), (
        head_dim_qk,
        head_dim_v,
    ), mask, learnable_sink, pack_gqa, num_splits, softcap in product(
        ADVANCED_RAGGED_SEQUENCES,
        ((40, 8), (8, 1)),
        HEAD_DIMS,
        (None, "causal", (44, 88)),
        (False, True),
        (False, True),
        (-1, 0, 5),
        (0.0, 50.0),
    )
]

PAGED_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.KVCACHE,
        q_layout=FmhaLayout.RAGGED,
        kv_layout=FmhaLayout.PAGED,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        page_size=page_size,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        cp_world_size=cp_world_size,
        compare_lse=cp_world_size == 1,
        **_sequence_options(sequence),
        **_mask_options(mask),
    )
    for sequence, (num_heads_q, num_heads_kv), (
        head_dim_qk,
        head_dim_v,
    ), page_size, mask, pack_gqa, num_splits, cp_world_size in product(
        PAGED_SEQUENCES,
        HEADS,
        HEAD_DIMS,
        (1, 3, 4, 16, 64, 111),
        (None, "causal"),
        (True, False),
        (0, 5),
        (1, 2, 4),
    )
]

PAGED_ADVANCED_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.KVCACHE,
        q_layout=FmhaLayout.RAGGED if varlen_q else FmhaLayout.NORMAL,
        kv_layout=FmhaLayout.PAGED,
        batch_size=3,
        seqlen_q=222,
        seqlen_k=1328,
        q_lengths=(111, 55, 222),
        kv_lengths=(1328, 666, 463),
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        page_size=page_size,
        learnable_sink=learnable_sink,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
        **_mask_options(mask),
    )
    for (num_heads_q, num_heads_kv), (
        head_dim_qk,
        head_dim_v,
    ), page_size, mask, varlen_q, learnable_sink, pack_gqa, num_splits in product(
        ((40, 8), (8, 1)),
        HEAD_DIMS,
        (1, 4, 16, 64),
        (None, "causal", (44, 88)),
        (True, False),
        (True, False),
        (True, False),
        (0, 5),
    )
]

COMBINE_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.COMBINE,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        num_heads_q=num_heads_q,
        head_dim_v=head_dim_v,
        q_layout=layout,
        num_splits=5,
        randomize_splits=True,
    )
    for batch_size, layout, num_heads_q, head_dim_v, seqlen_q in product(
        (1, 5, 32),
        (FmhaLayout.NORMAL, FmhaLayout.RAGGED, FmhaLayout.PADDED),
        (1, 4, 5, 8),
        (128, 256),
        (1, 3),
    )
] + [
    FmhaWorkload(
        api=FmhaApi.COMBINE,
        batch_size=2,
        seqlen_q=2,
        num_heads_q=2,
        head_dim_v=64,
        num_splits=num_splits,
        reuse_output=reuse_output,
    )
    for num_splits, reuse_output in product((129, 256), (False, True))
]

ADVANCED_WORKLOADS = [
    FmhaWorkload(
        api=FmhaApi.KVCACHE,
        input_mode=FmhaInputMode.ADVANCED,
        verify_mode=FmhaVerifyMode.ADVANCED,
        batch_size=batch_size,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        num_heads_q=96,
        num_heads_kv=8,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        dtype=dtype,
        q_layout=FmhaLayout.RAGGED,
        kv_layout=(FmhaLayout.NORMAL if page_size is None else FmhaLayout.PAGED),
        page_size=page_size,
        has_qv=has_qv,
        only_qv=only_qv,
        new_kv=True,
        has_cache_batch_idx=True,
        has_cache_leftpad=True,
        rotary_fraction=0.5,
        rotary_interleaved=True,
        has_rotary_seqlens=True,
        causal=True,
        attention_chunk=65,
        softcap=50.0,
        pack_gqa=True,
        num_splits=0,
    )
    for dtype, page_size, (head_dim_qk, head_dim_v), (has_qv, only_qv), (
        seqlen_q,
        seqlen_k,
    ), batch_size in product(
        (torch.bfloat16, torch.float8_e4m3fn),
        (None, 32, 64),
        ((192, 128), (128, 128), (64, 256), (64, 512)),
        ((False, False), (True, False), (True, True)),
        ((1, 339), (3, 1024), (64, 800), (3, 799), (16, 20000)),
        (1, 43, 77),
    )
]


def _run_fmha_workload(workload: FmhaWorkload) -> None:
    operator = FmhaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL

    with operator.use_mode(mode):
        try:
            inputs = operator.generate(workload)
        except UnsupportedFmhaWorkload as error:
            pytest.skip(str(error))
        reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
        outputs = operator.call(inputs)
        if reference is not None:
            operator.verify(inputs, outputs, reference)

    if workload.input_mode is FmhaInputMode.ADVANCED:
        del inputs, outputs, reference
        gc.collect()
        torch.musa.empty_cache()


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", METADATA_WORKLOADS, ids=str)
@torch.inference_mode()
def test_metadata(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", STRIDE_WORKLOADS[:2], ids=str)
@torch.inference_mode()
def test_varlen_mubin_preserves_noncontiguous_k_stride(
    workload: FmhaWorkload,
) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", STRIDE_WORKLOADS[2:], ids=str)
@torch.inference_mode()
def test_mubin_accepts_large_singleton_batch_strides(
    workload: FmhaWorkload,
) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", DENSE_WORKLOADS, ids=str)
@torch.inference_mode()
def test_varlen_func_bshd(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", FP8_SOFTCAP_ALL_MASKED_WORKLOADS, ids=str)
@torch.inference_mode()
def test_fp8_softcap_all_masked_block(workload: FmhaWorkload) -> None:
    operator = FmhaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL

    with operator.use_mode(mode):
        inputs = operator.generate(replace(workload, dtype=workload.reference_dtype))
        assert inputs.q_padded is not None
        assert inputs.k_padded is not None
        assert inputs.v_padded is not None
        assert workload.num_heads_kv is not None

        fp8_dtype = workload.dtype
        q_ref = (
            (inputs.q_padded * workload.softcap / 4)
            .to(fp8_dtype)
            .to(workload.reference_dtype)
        )
        k_ref = inputs.k_padded.to(fp8_dtype).to(workload.reference_dtype)
        v_ref = inputs.v_padded.to(fp8_dtype).to(workload.reference_dtype)
        inputs.q_padded, inputs.k_padded, inputs.v_padded = q_ref, k_ref, v_ref
        inputs.q, inputs.k, inputs.v = (
            q_ref.to(fp8_dtype),
            k_ref.to(fp8_dtype),
            v_ref.to(fp8_dtype),
        )
        inputs.q_descale, inputs.k_descale, inputs.v_descale = [
            torch.rand(
                workload.batch_size,
                workload.num_heads_kv,
                device=workload.device,
                dtype=torch.float32,
            )
            * 2
            for _ in range(3)
        ]
        inputs.workload = workload

        reference = None
        if mode is OpMode.NORMAL:
            reference_args = dict(
                q=q_ref,
                k=k_ref,
                v=v_ref,
                causal=workload.causal,
                q_descale=inputs.q_descale,
                k_descale=inputs.k_descale,
                v_descale=inputs.v_descale,
                window_size=workload.window_size,
                attention_chunk=workload.attention_chunk,
                softcap=workload.softcap,
            )
            out_ref, _, _ = attention_ref(**reference_args)
            baseline_out, _, _ = attention_ref(
                **reference_args,
                upcast=False,
                reorder_ops=True,
                intermediate_dtype=fp8_dtype,
            )
            reference = FmhaReference(out=out_ref, baseline_out=baseline_out)

        outputs = operator.call(inputs)
        if reference is not None:
            operator.verify(inputs, outputs, reference)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", RAGGED_WORKLOADS, ids=str)
@torch.inference_mode()
def test_varlen_func_ragged_qkv(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", RAGGED_ADVANCED_WORKLOADS, ids=str)
@torch.inference_mode()
def test_varlen_func_ragged_qkv_advance(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", PAGED_WORKLOADS, ids=str)
@torch.inference_mode()
def test_paged_attn(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", PAGED_ADVANCED_WORKLOADS, ids=str)
@torch.inference_mode()
def test_paged_attn_advance(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", COMBINE_WORKLOADS[:-4], ids=str)
@torch.inference_mode()
def test_combine(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", COMBINE_WORKLOADS[-4:], ids=str)
@torch.inference_mode()
def test_combine_high_splits(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", ADVANCED_WORKLOADS, ids=str)
@torch.inference_mode()
def test_advance_features(workload: FmhaWorkload) -> None:
    _run_fmha_workload(workload)
