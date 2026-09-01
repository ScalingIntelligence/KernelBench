.. _apisparsemla:

Sparse MLA
==========

Use the :doc:`FlashInfer wrapper <../wrappers/flashinfer_wrapper>` when your
framework expects the ``flashinfer.rope`` and ``flashinfer.decode`` package
surface. The native APIs below expose MATE's fused MLA RoPE quantization and
FP8 sparse decode path directly.

RoPE and FP8 quantization
-------------------------

.. currentmodule:: mate.sparse_mla_interface

``mla_rope_quantize_fp8`` applies rotary embedding to the 64-value RoPE tail,
quantizes the 512-value latent component and the rotated tail to FP8 E4M3, and
supports writing both components into slices of caller-owned 576-byte rows.

.. autofunction:: mla_rope_quantize_fp8

FP8 sparse decode
-----------------

Use ``mate.sparse_mla_interface.get_batch_decode_metadata_mla`` to prepare
scheduler metadata once while the query shape, sparse lengths, and top-k remain
unchanged, then pass it to subsequent decode calls.

.. autofunction:: sparse_mla_fp8_decode
