.. _apiflashinfer:

FlashInfer Wrapper APIs
=======================

This page documents the supported ``flashinfer`` compatibility functions.
See the :doc:`FlashInfer wrapper guide <../wrappers/flashinfer_wrapper>` for
installation, supported workflows, constraints, and the end-to-end FP8 MLA
example.

GEMM
----

.. currentmodule:: flashinfer.gemm

.. autofunction:: bmm_bf16
.. autofunction:: bmm_fp8
.. autofunction:: gemm_fp8_nt_groupwise
.. autofunction:: group_deepgemm_fp8_nt_groupwise
.. autofunction:: batch_deepgemm_fp8_nt_groupwise

FP8 MLA RoPE quantization
-------------------------

.. currentmodule:: flashinfer.rope

.. autofunction:: mla_rope_quantize_fp8

Sparse MLA decode metadata
--------------------------

.. currentmodule:: flashinfer.decode

Prepare metadata once and reuse it only while the query shape, sparse lengths,
and top-k remain unchanged.

.. autofunction:: get_batch_decode_metadata_mla

``query`` must be a MUSA tensor with shape
``[batch, q_len, num_heads, 576]``. ``seq_lens`` must be a contiguous int32
tensor with shape ``[batch]`` (or ``[batch, 1]``) on the same device.
``sparse_mla_top_k`` must be a positive multiple of 64.

The return value is opaque scheduler metadata. Pass it to
``trtllm_batch_decode_with_kv_cache_mla`` through ``metadata``. Prepare new
metadata after changing the query shape or device, the ``seq_lens`` values, or
the sparse top-k, and do not share one metadata object between overlapping
decode calls.

FP8 sparse MLA decode
---------------------

.. autofunction:: trtllm_batch_decode_with_kv_cache_mla
