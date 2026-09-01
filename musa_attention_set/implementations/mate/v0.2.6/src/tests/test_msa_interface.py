from __future__ import annotations

import ast
import inspect

import pytest
import torch

import mate
from mate import msa_interface as msa
from mate.execution_context import is_dry_run_enabled
from mate.testing import supported_musa_compute_capability
from mate.testing.operator import OpMode
from mate.testing.operator.msa import (
    MsaApi,
    MsaKernelMode,
    MsaKvLayout,
    MsaOperator,
    MsaWorkload,
    UnsupportedMsaWorkload,
)

_HAS_FP8_E4M3 = hasattr(torch, "float8_e4m3fn")
_FWD_DTYPES = [torch.float16, torch.bfloat16]
if _HAS_FP8_E4M3:
    _FWD_DTYPES.append(torch.float8_e4m3fn)


def test_msa_maxscore_prefill_tile_q_dispatch():
    from mate.jit.msa_ops import _maxscore_tile_q

    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.bfloat16,
            max_seqlen_q=64 * 1024,
        )
        == 128
    )
    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.bfloat16,
            max_seqlen_q=32 * 1024,
        )
        == 16
    )
    assert _maxscore_tile_q(
        1,
        True,
        dtype=torch.float8_e4m3fn if _HAS_FP8_E4M3 else torch.bfloat16,
        max_seqlen_q=128 * 1024,
    ) == (_HAS_FP8_E4M3 and 64 or 128)


def test_msa_plan_from_lengths_dense_contract():
    qo_lens = torch.tensor([8, 16], dtype=torch.int64)
    kv_lens = torch.tensor([32, 64], dtype=torch.int64)

    has_mixed, split, batch_size, plan, prefill = msa._msa_plan_from_lengths(
        qo_lens.cpu(),
        kv_lens.cpu(),
        16,
        num_kv_heads=4,
        causal=True,
    )

    assert has_mixed is False
    assert split == 0
    assert batch_size == 2
    assert prefill is None
    assert plan.mode == "dense"
    assert plan.num_qo_heads == 16
    assert plan.num_kv_heads == 4
    assert plan.kv_page_indptr is None
    assert plan.prefill_plan.total_seqlen_k == 96
    assert torch.equal(
        plan.qo_offset, kv_lens.to(torch.int32) - qo_lens.to(torch.int32)
    )
    assert torch.equal(
        plan.prefill_plan.cu_seqlens_q,
        torch.tensor([0, 8, 24], dtype=torch.int32),
    )


def test_msa_sparse_prefill_mode_is_declared():
    qo_lens = torch.tensor([64], dtype=torch.int32)
    kv_lens = torch.tensor([256], dtype=torch.int32)

    _, _, _, plan, _ = msa._msa_plan_from_lengths(
        qo_lens.cpu(),
        kv_lens.cpu(),
        16,
        num_kv_heads=4,
        page_size=128,
        kv_block_num=16,
    )

    assert plan.mode == "sparse_prefill"
    assert plan.prefill_plan.total_seqlen_k == 256
    assert torch.equal(
        plan.kv_page_indptr,
        torch.tensor([0, 2], dtype=torch.int32),
    )


def test_msa_plan_from_lengths_keeps_topk_separate_from_selected_block_count():
    _, _, _, plan, _ = msa._msa_plan_from_lengths(
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([4096], dtype=torch.int32),
        8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
    )

    assert plan.kv_block_num == 16
    assert plan.selected_block_count == 18
    assert plan.force_begin_blocks == 1
    assert plan.force_end_blocks == 1
    assert not plan.force_blocks_count_in_topk


def test_msa_plan_requires_device_runtime_metadata():
    plan_info = msa.msa_plan(
        batch_size=1,
        max_seqlen_q=128,
        max_seqlen_k=4096,
        num_qo_heads=8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        sparse_kernel_mode="prefill",
    )
    _, _, _, plan, extra = plan_info
    assert extra is None
    assert plan.is_static
    assert plan.mode == "sparse_prefill"
    assert plan.prefill_plan.max_seqlen_q == 128
    assert plan.prefill_plan.max_seqlen_k == 4096
    assert torch.count_nonzero(plan.qo_lens) == 0

    q = torch.empty((1, 8, 128), dtype=torch.float16)
    k = torch.empty((32, 1, 128, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indexes = torch.zeros((1, 1, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match="static MSA plans require runtime_metadata"):
        msa.sparse_msa(
            q,
            k,
            v,
            plan_info,
            kv_indices=torch.arange(32, dtype=torch.int32),
            kv_block_indexes=block_indexes,
        )


def test_msa_plan_is_the_only_public_capacity_planner():
    assert "msa_plan" in msa.__all__
    assert "msa_plan_static" not in msa.__all__
    assert "msa_plan" in mate.__all__
    assert "msa_plan_static" not in mate.__all__
    assert not hasattr(msa, "msa_plan_static")
    assert not hasattr(mate, "msa_plan_static")


def test_msa_interface_does_not_use_tensor_item():
    tree = ast.parse(inspect.getsource(msa))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    assert offenders == []


def test_eager_msa_plan_rejects_accelerator_lengths():
    qo_lens = torch.ones((1,), dtype=torch.int32, device="meta")
    kv_lens = torch.ones((1,), dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match="must be a CPU tensor"):
        msa._msa_plan_from_lengths(qo_lens, kv_lens, 8, num_kv_heads=1)


def test_msa_static_runtime_metadata_reaches_fwd_kernel(monkeypatch):
    calls = []

    def fake_msa_fwd(q, k, v, block_indexes, *args, **kwargs):
        calls.append((q, k, v, block_indexes, args, kwargs))
        return torch.empty_like(q), torch.empty((q.shape[0], q.shape[1]))

    monkeypatch.setattr(msa, "_msa_fwd", fake_msa_fwd)
    plan_info = msa.msa_plan(
        batch_size=1,
        max_seqlen_q=128,
        max_seqlen_k=4096,
        num_qo_heads=8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        sparse_kernel_mode="prefill",
    )
    runtime = msa.MsaRuntimeMetadata(
        qo_lens=torch.tensor([1], dtype=torch.int32),
        kv_lens=torch.tensor([256], dtype=torch.int32),
        qo_offset=torch.tensor([255], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 256], dtype=torch.int32),
        seqused_k=torch.tensor([256], dtype=torch.int32),
        page_table=torch.cat(
            [
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.zeros((1, 30), dtype=torch.int32),
            ],
            dim=1,
        ),
    )
    q = torch.empty((1, 8, 128), dtype=torch.float16)
    k = torch.empty((2, 1, 128, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indexes = torch.zeros((1, 1, 16), dtype=torch.int32)
    lse_buffer = torch.empty((1, 8), dtype=torch.float32)
    out, lse = msa.sparse_msa(
        q,
        k,
        v,
        plan_info,
        kv_block_indexes=block_indexes,
        lse=lse_buffer,
        runtime_metadata=runtime,
    )
    assert out.shape == q.shape
    assert lse is None
    assert len(calls) == 1
    # args: cu_seqlens_q, seqused_k, qo_offset, page_table, ...
    assert calls[0][4][0] is runtime.cu_seqlens_q
    assert calls[0][4][1] is runtime.seqused_k
    assert calls[0][4][2] is runtime.qo_offset
    assert calls[0][4][3] is runtime.page_table
    assert calls[0][5]["kv_page_indptr"] is None
    assert calls[0][5]["lse"] is lse_buffer


def test_msa_static_runtime_metadata_reaches_paged_maxscore(monkeypatch):
    calls = []

    def fake_msa_maxscore(q, k, cu_q, cu_k, qo_offset, **kwargs):
        calls.append((q, k, cu_q, cu_k, qo_offset, kwargs))
        return kwargs["max_score"]

    monkeypatch.setattr(msa, "_msa_maxscore", fake_msa_maxscore)
    plan_info = msa.msa_plan(
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        total_seqlen_k=8192,
        num_qo_heads=1,
        num_kv_heads=1,
        page_size=128,
        output_maxscore=True,
    )
    _, _, _, plan, _ = plan_info
    assert plan.prefill_plan.total_seqlen_k == 8192

    runtime = msa.MsaRuntimeMetadata(
        qo_lens=torch.ones((2,), dtype=torch.int32),
        kv_lens=torch.tensor([256, 384], dtype=torch.int32),
        qo_offset=torch.tensor([255, 383], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 256, 640], dtype=torch.int32),
        seqused_k=torch.tensor([256, 384], dtype=torch.int32),
        page_table=torch.zeros((2, 32), dtype=torch.int32),
    )
    q = torch.empty((2, 1, 128), dtype=torch.float16)
    k = torch.empty((64, 128, 1, 128), dtype=torch.float16)
    max_score = torch.empty((2, 1, 128), dtype=torch.float32)
    out, actual = msa.msa(
        q,
        k,
        k,
        plan_info,
        output_o=False,
        output_maxscore=True,
        max_score=max_score,
        runtime_metadata=runtime,
    )

    assert out is None
    assert actual is max_score
    assert len(calls) == 1
    assert calls[0][2] is runtime.cu_seqlens_q
    assert calls[0][3] is runtime.cu_seqlens_k
    assert calls[0][4] is runtime.qo_offset
    assert calls[0][5]["page_table"] is runtime.page_table
    assert calls[0][5]["kv_page_indptr"] is None


def test_sparse_msa_plan_rejects_page_block_size_mismatch():
    with pytest.raises(ValueError, match="page_size == sparse_block_size"):
        msa.sparse_msa_plan(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            sparse_block_size=128,
            kv_block_num=4,
            causal=True,
        )


def test_msa_plan_from_lengths_rejects_page64_paged_maxscore():
    with pytest.raises(ValueError, match="paged MSA maxscore requires page_size"):
        msa._msa_plan_from_lengths(
            torch.tensor([64], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            output_maxscore=True,
            causal=True,
        )


def test_sparse_msa_plan_rejects_unsupported_sparse_block_size():
    with pytest.raises(ValueError, match="sparse_block_size == 128"):
        msa.sparse_msa_plan(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            sparse_block_size=64,
            kv_block_num=4,
            causal=True,
        )


def test_build_page_table_from_flat_kv_indices():
    kv_lens = torch.tensor([256, 384], dtype=torch.int32)
    kv_indices = torch.tensor([3, 5, 7, 11, 13], dtype=torch.int32)

    page_table = msa._build_page_table_from_flat_kv_indices(
        kv_indices,
        kv_lens,
        page_size=128,
    )

    assert torch.equal(
        page_table,
        torch.tensor([[3, 5, 0], [7, 11, 13]], dtype=torch.int32),
    )


@pytest.mark.parametrize("mode,q_len", [("prefill", 4), ("decode", 1)])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (16, 1), (32, 2)])
@pytest.mark.parametrize("dtype", _FWD_DTYPES)
def test_sparse_modes_route_to_unified_msa_fwd(
    monkeypatch,
    mode: str,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
):
    calls = []

    def fake_msa_fwd(q, k, v, block_indexes, *args, **kwargs):
        calls.append((q, k, v, block_indexes, args, kwargs))
        return (
            torch.empty_like(q),
            torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32),
        )

    monkeypatch.setattr(msa, "_msa_fwd", fake_msa_fwd)
    qo_lens = torch.tensor([q_len], dtype=torch.int32)
    kv_lens = torch.tensor([256], dtype=torch.int32)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens.cpu(),
        kv_lens.cpu(),
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=128,
        sparse_block_size=128,
        kv_block_num=16,
        sparse_kernel_mode=mode,
        split_prefill_decode=False,
        causal=False,
    )
    q = torch.empty((q_len, num_q_heads, 128), dtype=dtype)
    k = torch.empty((2, num_kv_heads, 128, 128), dtype=dtype)
    v = torch.empty_like(k)
    block_indexes = torch.full((q_len, num_kv_heads, 16), -1, dtype=torch.int32)
    block_indexes[..., :2] = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.arange(2, dtype=torch.int32)

    if mode == "prefill":
        out, lse = msa.sparse_msa(
            q,
            k,
            v,
            plan_info,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            k_scale=1.25,
            v_scale=1.5,
        )
        assert lse is None
    else:
        out = msa.sparse_decode_atten_func(
            q,
            k,
            v,
            plan_info,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            k_scale=1.25,
            v_scale=1.5,
        )

    assert out.shape == q.shape
    assert len(calls) == 1
    assert calls[0][3] is block_indexes
    assert calls[0][5]["k_scale"] == 1.25
    assert calls[0][5]["v_scale"] == 1.5


_MAXSCORE_WORKLOADS = [
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=4,
        num_kv_heads=2,
        q_lengths=(3, 5),
        kv_lengths=(140, 65),
        kv_layout=MsaKvLayout.DENSE,
        dtype=torch.float16,
        causal=True,
        seed=301,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=4,
        num_kv_heads=2,
        q_lengths=(2, 4),
        kv_lengths=(130, 70),
        kv_layout=MsaKvLayout.PAGED,
        page_mode="reverse",
        dtype=torch.float16,
        causal=True,
        preallocated_max_score=True,
        seed=302,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=8,
        num_kv_heads=2,
        q_lengths=(33, 64),
        kv_lengths=(257, 130),
        kv_layout=MsaKvLayout.DENSE,
        dtype=torch.bfloat16,
        causal=True,
        seed=304,
    ),
    MsaWorkload(
        api=MsaApi.MAXSCORE,
        num_q_heads=8,
        num_kv_heads=2,
        q_lengths=(33, 64),
        kv_lengths=(257, 130),
        kv_layout=MsaKvLayout.PAGED,
        page_mode="reverse",
        dtype=torch.bfloat16,
        causal=True,
        seed=304,
    ),
]
if _HAS_FP8_E4M3:
    _MAXSCORE_WORKLOADS.extend(
        [
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=8,
                num_kv_heads=2,
                q_lengths=(33, 64),
                kv_lengths=(257, 130),
                kv_layout=MsaKvLayout.DENSE,
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=303,
            ),
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=8,
                num_kv_heads=2,
                q_lengths=(33, 64),
                kv_lengths=(257, 130),
                kv_layout=MsaKvLayout.PAGED,
                page_mode="reverse",
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=303,
            ),
            MsaWorkload(
                api=MsaApi.MAXSCORE,
                num_q_heads=16,
                num_kv_heads=1,
                q_lengths=(1, 1),
                kv_lengths=(385, 130),
                kv_layout=MsaKvLayout.PAGED,
                page_mode="reverse",
                dtype=torch.float8_e4m3fn,
                causal=True,
                seed=305,
            ),
        ]
    )


def _sparse_topk_workloads() -> list[MsaWorkload]:
    workloads = [
        MsaWorkload(
            api=MsaApi.SPARSE_TOPK,
            num_q_heads=17,
            num_kv_heads=1,
            total_q=3,
            max_k_tiles=128,
            topk=topk,
            num_valid_pages=91,
            spike_indices=(19, 57),
            spike_values=(8.0, 6.0),
            invalid_after=91,
            seed=401 + topk,
        )
        for topk in (4, 8, 16)
    ]
    workloads.append(
        MsaWorkload(
            api=MsaApi.SPARSE_TOPK,
            num_q_heads=9,
            num_kv_heads=1,
            total_q=2,
            max_k_tiles=128,
            topk=8,
            num_valid_pages=126,
            dip_indices=(3, 125),
            invalid_after=126,
            force_begin_blocks=4,
            force_end_blocks=3,
            preallocated_output=True,
            seed=402,
        )
    )
    for topk, (max_k_tiles, num_valid_pages) in (
        (4, (128, 126)),
        (8, (128, 126)),
        (16, (128, 126)),
        (4, (512, 500)),
        (8, (512, 500)),
        (16, (512, 500)),
    ):
        workloads.append(
            MsaWorkload(
                api=MsaApi.SPARSE_TOPK,
                num_q_heads=3,
                num_kv_heads=1,
                total_q=2,
                max_k_tiles=max_k_tiles,
                topk=topk,
                num_valid_pages=num_valid_pages,
                dip_indices=(0, num_valid_pages - 1),
                dip_value=-1000.0,
                invalid_after=num_valid_pages,
                force_begin_blocks=1,
                force_end_blocks=1,
                force_blocks_count_in_topk=False,
                seed=403 + max_k_tiles,
            )
        )
    workloads.append(
        MsaWorkload(
            api=MsaApi.SPARSE_TOPK,
            num_q_heads=5,
            num_kv_heads=1,
            total_q=1,
            max_k_tiles=8,
            topk=8,
            num_valid_pages=5,
            score_pattern="zeros",
            seed=404,
        )
    )
    for max_k_tiles, num_valid_pages in ((128, 96), (512, 500)):
        workloads.append(
            MsaWorkload(
                api=MsaApi.SPARSE_TOPK,
                num_q_heads=3,
                num_kv_heads=1,
                total_q=4,
                max_k_tiles=max_k_tiles,
                topk=16,
                num_valid_pages=num_valid_pages,
                force_begin_blocks=1,
                force_end_blocks=1,
                force_blocks_count_in_topk=False,
                query_positions=(0, 129, 2303, 6144),
                seed=405,
            )
        )
    return workloads


_SPARSE_TOPK_WORKLOADS = _sparse_topk_workloads()


_SPARSE_FWD_WORKLOADS = [
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(64,),
        kv_lengths=(128,),
        dtype=torch.float16,
        causal=True,
        pattern="first",
        page_mode="identity",
        seed=501,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(48, 64),
        kv_lengths=(128, 192),
        dtype=torch.float16,
        causal=True,
        pattern="rolling",
        page_mode="reverse",
        seed=502,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=16,
        num_kv_heads=2,
        q_lengths=(64,),
        kv_lengths=(256,),
        dtype=torch.float16,
        causal=True,
        pattern="tail",
        page_mode="identity",
        seed=503,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=16,
        num_kv_heads=2,
        q_lengths=(64,),
        kv_lengths=(256,),
        dtype=torch.float16,
        causal=False,
        pattern="rolling",
        page_mode="reverse",
        seed=504,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.PREFILL,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(64,),
        kv_lengths=(192,),
        dtype=torch.bfloat16,
        causal=True,
        pattern="rolling",
        page_mode="identity",
        seed=505,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(1,),
        kv_lengths=(128,),
        dtype=torch.float16,
        causal=True,
        pattern="first",
        page_mode="identity",
        seed=506,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=16,
        num_kv_heads=1,
        q_lengths=(4,),
        kv_lengths=(256,),
        dtype=torch.float16,
        causal=True,
        pattern="tail",
        page_mode="reverse",
        seed=507,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=16,
        num_kv_heads=2,
        q_lengths=(2, 4),
        kv_lengths=(192, 256),
        dtype=torch.float16,
        causal=True,
        pattern="rolling",
        page_mode="identity",
        seed=508,
    ),
    MsaWorkload(
        api=MsaApi.SPARSE_FWD,
        kernel_mode=MsaKernelMode.DECODE,
        num_q_heads=8,
        num_kv_heads=1,
        q_lengths=(1, 4),
        kv_lengths=(256, 128),
        dtype=torch.bfloat16,
        causal=True,
        pattern="rolling",
        page_mode="reverse",
        seed=509,
    ),
]


def _run_msa_workload(workload: MsaWorkload) -> None:
    operator = MsaOperator()
    mode = OpMode.DRY_RUN if is_dry_run_enabled() else OpMode.NORMAL

    with operator.use_mode(mode):
        try:
            inputs = operator.generate(workload)
        except UnsupportedMsaWorkload as error:
            pytest.skip(str(error))
        reference = operator.reference(inputs) if mode is OpMode.NORMAL else None
        outputs = operator.call(inputs)
        if reference is not None:
            operator.verify(inputs, outputs, reference)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", _MAXSCORE_WORKLOADS, ids=str)
@torch.inference_mode()
def test_msa_output_maxscore_matches_reference(workload: MsaWorkload) -> None:
    _run_msa_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", _SPARSE_TOPK_WORKLOADS, ids=str)
@torch.inference_mode()
def test_msa_sparse_topk_select_matches_reference(workload: MsaWorkload) -> None:
    _run_msa_workload(workload)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("workload", _SPARSE_FWD_WORKLOADS, ids=str)
@torch.inference_mode()
def test_msa_sparse_forward_matches_reference(workload: MsaWorkload) -> None:
    _run_msa_workload(workload)
