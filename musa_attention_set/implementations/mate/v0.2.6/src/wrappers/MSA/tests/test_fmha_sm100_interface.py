from __future__ import annotations

import ast
import inspect
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fmha_sm100
from fmha_sm100 import sparse as fmha_sparse
from mate import msa_interface as mate_msa

_HAS_MUSA = hasattr(torch, "musa") and torch.musa.is_available()


def test_sparse_wrapper_does_not_use_tensor_item():
    tree = ast.parse(inspect.getsource(fmha_sparse))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    ]
    assert offenders == []


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for sparse decode")
def test_sparse_decode_static_runtime_matches_eager_plan():
    device = torch.device("musa")
    torch.manual_seed(17)
    q = torch.randn((1, 8, 128), device=device, dtype=torch.float16) * 0.1
    k = torch.randn((16, 1, 128, 128), device=device, dtype=torch.float16) * 0.1
    v = torch.randn_like(k) * 0.1
    page_table = torch.arange(16, device=device, dtype=torch.int32).view(1, 16)
    seqused_k = torch.tensor([2048], device=device, dtype=torch.int32)
    q2k = torch.arange(16, device=device, dtype=torch.int32).view(1, 1, 16)

    actual = fmha_sparse.sparse_decode_atten_func(
        q,
        k,
        v,
        q2k,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=1,
        max_seqlen_k=2048,
    )

    eager_plan = mate_msa._msa_plan_from_lengths(
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2048], dtype=torch.int32),
        8,
        num_kv_heads=1,
        page_size=128,
        sparse_block_size=128,
        kv_block_num=16,
        sparse_kernel_mode="decode",
        split_prefill_decode=False,
    )
    expected = mate_msa.sparse_decode_atten_func(
        q,
        k,
        v,
        eager_plan,
        page_table=page_table,
        kv_block_indexes=q2k.permute(1, 0, 2).contiguous(),
    )
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


def test_root_plan_and_run_forward_to_mate(monkeypatch):
    captured = {}

    def fake_plan(*args, **kwargs):
        captured["plan"] = (args, kwargs)
        return "plan-info"

    def fake_run(*args, **kwargs):
        captured["run"] = (args, kwargs)
        return "out"

    monkeypatch.setattr(mate_msa, "_msa_plan_from_lengths", fake_plan)
    monkeypatch.setattr(mate_msa, "msa", fake_run)

    assert fmha_sm100.fmha_sm100_plan("q", num_kv_heads=1) == "plan-info"
    assert fmha_sm100.fmha_sm100("q", "k", "v", plan_info="plan-info") == "out"
    assert captured["plan"] == (("q",), {"num_kv_heads": 1})
    assert captured["run"] == (("q", "k", "v"), {"plan_info": "plan-info"})


def test_sparse_fwd_symbols_forward_to_mate(monkeypatch):
    monkeypatch.setattr(mate_msa, "sparse_msa_plan", lambda *a, **k: ("plan", a, k))
    monkeypatch.setattr(mate_msa, "sparse_msa", lambda *a, **k: ("fmha", a, k))

    assert fmha_sparse.sparse_fmha_plan(1, x=2)[0] == "plan"
    assert fmha_sparse.sparse_fmha(1, x=2)[0] == "fmha"


def test_sparse_decode_msa_signature_routes_to_mate_decode(monkeypatch):
    captured = {}
    plan_sentinel = object()
    q = torch.empty((4, 8, 128), dtype=torch.float16)
    k = torch.empty((8, 1, 128, 128), dtype=torch.float16)
    v = torch.empty((8, 1, 128, 128), dtype=torch.float16)
    q2k = torch.zeros((1, 4, 16), dtype=torch.int32)
    page_table = torch.zeros((1, 8), dtype=torch.int32)
    seqused_k = torch.tensor([128], dtype=torch.int32)
    lse = torch.empty((8, 4), dtype=torch.float32)

    def fake_plan(*args, **kwargs):
        captured["plan_args"] = args
        captured["plan_kwargs"] = kwargs
        return plan_sentinel

    def fake_decode(*args, **kwargs):
        captured["decode_args"] = args
        captured["decode_kwargs"] = kwargs
        return torch.empty_like(q), lse

    monkeypatch.setattr(mate_msa, "msa_plan", fake_plan)
    monkeypatch.setattr(mate_msa, "sparse_decode_atten_func", fake_decode)

    out, returned_lse = fmha_sparse.sparse_decode_atten_func(
        q,
        k,
        v,
        q2k,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=4,
        max_seqlen_k=1024,
        return_softmax_lse=True,
    )

    assert out.shape == q.shape
    assert returned_lse.shape == (4, 8)
    assert captured["plan_args"] == ()
    assert captured["plan_kwargs"]["batch_size"] == 1
    assert captured["plan_kwargs"]["max_seqlen_q"] == 4
    assert captured["plan_kwargs"]["max_seqlen_k"] == 1024
    assert captured["plan_kwargs"]["total_seqlen_k"] == 1024
    assert captured["plan_kwargs"]["kv_block_num"] == 16
    assert captured["plan_kwargs"]["sparse_kernel_mode"] == "decode"
    assert captured["decode_args"][3] is plan_sentinel
    runtime = captured["decode_kwargs"]["runtime_metadata"]
    assert isinstance(runtime, mate_msa.MsaRuntimeMetadata)
    assert runtime.page_table is page_table
    assert runtime.kv_lens is seqused_k
    assert "kv_indices" not in captured["decode_kwargs"]
    torch.testing.assert_close(
        captured["decode_kwargs"]["kv_block_indexes"],
        q2k.permute(1, 0, 2).contiguous(),
    )
    assert captured["decode_kwargs"]["return_softmax_lse"] is True


def test_sparse_decode_none_q2k_routes_to_dense_paged_decode(monkeypatch):
    captured = {}
    plan_sentinel = object()
    q = torch.empty((4, 8, 128), dtype=torch.float16)
    k = torch.empty((1, 1, 128, 128), dtype=torch.float16)
    v = torch.empty((1, 1, 128, 128), dtype=torch.float16)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    seqused_k = torch.tensor([128], dtype=torch.int32)

    def fake_plan(*args, **kwargs):
        captured["plan_args"] = args
        captured["plan_kwargs"] = kwargs
        return plan_sentinel

    def fake_msa(*args, **kwargs):
        captured["msa_args"] = args
        captured["msa_kwargs"] = kwargs
        return "dense-decode", None

    def fail_sparse_decode(*args, **kwargs):
        raise AssertionError("q2k_indices=None should not route to sparse decode")

    monkeypatch.setattr(mate_msa, "msa_plan", fake_plan)
    monkeypatch.setattr(mate_msa, "msa", fake_msa)
    monkeypatch.setattr(mate_msa, "sparse_decode_atten_func", fail_sparse_decode)

    result = fmha_sparse.sparse_decode_atten_func(
        q,
        k,
        v,
        None,
        page_table=page_table,
        seqused_k=seqused_k,
        seqlen_q=4,
        max_seqlen_k=128,
        softmax_scale=0.25,
    )

    assert result == "dense-decode"
    assert captured["plan_args"] == ()
    assert captured["plan_kwargs"]["batch_size"] == 1
    assert captured["plan_kwargs"]["max_seqlen_q"] == 4
    assert captured["plan_kwargs"]["max_seqlen_k"] == 128
    assert captured["plan_kwargs"]["total_seqlen_k"] == 128
    assert captured["plan_kwargs"]["page_size"] == 128
    assert captured["plan_kwargs"]["num_kv_splits"] == 1
    assert captured["plan_kwargs"]["kv_block_num"] == -1
    assert captured["plan_kwargs"]["sparse_kernel_mode"] == "auto"
    assert captured["msa_args"] == (q, k, v, plan_sentinel)
    runtime = captured["msa_kwargs"]["runtime_metadata"]
    assert isinstance(runtime, mate_msa.MsaRuntimeMetadata)
    assert runtime.page_table is page_table
    assert runtime.kv_lens is seqused_k
    assert captured["msa_kwargs"]["sm_scale"] == 0.25

    with pytest.raises(NotImplementedError, match="return_softmax_lse"):
        fmha_sparse.sparse_decode_atten_func(
            q,
            k,
            v,
            None,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=4,
            max_seqlen_k=128,
            return_softmax_lse=True,
        )


def test_sparse_decode_rejects_non_topk16():
    q = torch.empty((4, 8, 128), dtype=torch.float16)
    k = torch.empty((1, 1, 128, 128), dtype=torch.float16)
    v = torch.empty((1, 1, 128, 128), dtype=torch.float16)
    q2k = torch.zeros((1, 4, 4), dtype=torch.int32)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    seqused_k = torch.tensor([128], dtype=torch.int32)

    with pytest.raises(ValueError, match="topK=16"):
        fmha_sparse.sparse_decode_atten_func(
            q,
            k,
            v,
            q2k,
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=4,
            max_seqlen_k=128,
        )


def test_sparse_decode_wrapper_plan_run(monkeypatch):
    captured = {}

    def fake_decode(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "decoded"

    monkeypatch.setattr(fmha_sparse, "sparse_decode_atten_func", fake_decode)

    wrapper = fmha_sparse.SparseDecodePagedAttentionWrapper()
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    seqused_k = torch.tensor([128], dtype=torch.int32)
    q2k = torch.zeros((1, 4, 16), dtype=torch.int32)

    assert (
        wrapper.plan(
            page_table=page_table,
            seqused_k=seqused_k,
            seqlen_q=4,
            max_seqlen_k=128,
            q2k_indices=q2k,
        )
        is wrapper
    )
    assert wrapper.run("q", "k", "v", softmax_scale=0.5) == "decoded"
    assert captured["args"] == ("q", "k", "v", q2k)
    assert captured["kwargs"]["page_table"] is page_table
    assert captured["kwargs"]["seqused_k"] is seqused_k
    assert captured["kwargs"]["softmax_scale"] == 0.5


def test_reserved_sparse_symbols_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        fmha_sparse.sparse_atten_nvfp4_kv_func()
    with pytest.raises(NotImplementedError):
        fmha_sparse.fp4_indexer_block_scores()
    with pytest.raises(NotImplementedError):
        fmha_sparse.SparseK2qCsrBuilderSm100()
