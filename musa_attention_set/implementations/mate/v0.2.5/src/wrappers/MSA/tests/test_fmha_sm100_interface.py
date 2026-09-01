from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fmha_sm100
from fmha_sm100 import sparse as fmha_sparse
from mate import msa_interface as mate_msa


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
    k = torch.empty((1, 1, 128, 128), dtype=torch.float16)
    v = torch.empty((1, 1, 128, 128), dtype=torch.float16)
    q2k = torch.zeros((1, 4, 16), dtype=torch.int32)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
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

    monkeypatch.setattr(mate_msa, "_msa_plan_from_lengths", fake_plan)
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
    assert captured["plan_args"][0].tolist() == [4]
    assert captured["plan_args"][1].tolist() == [128]
    assert captured["plan_kwargs"]["kv_block_num"] == 16
    assert captured["plan_kwargs"]["sparse_kernel_mode"] == "decode"
    assert captured["decode_args"][3] is plan_sentinel
    assert captured["decode_kwargs"]["page_table"] is page_table
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

    monkeypatch.setattr(mate_msa, "_msa_plan_from_lengths", fake_plan)
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
    assert captured["plan_args"][0].tolist() == [4]
    assert captured["plan_args"][1].tolist() == [128]
    assert captured["plan_kwargs"]["page_size"] == 128
    assert captured["plan_kwargs"]["num_kv_splits"] == 1
    assert "kv_block_num" not in captured["plan_kwargs"]
    assert "sparse_kernel_mode" not in captured["plan_kwargs"]
    assert captured["msa_args"] == (q, k, v, plan_sentinel)
    torch.testing.assert_close(
        captured["msa_kwargs"]["kv_indices"],
        page_table.reshape(-1).contiguous(),
    )
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
