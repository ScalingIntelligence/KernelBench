.. _apigemm:

GEMM
=========

For framework integrations that already target DeepGEMM style Python APIs,
prefer the ``deep-gemm`` wrapper first. Use the MATE APIs below when wrapper
coverage is not enough.

MoE GEMM
-------------------------

.. currentmodule:: mate.gemm

The direct MoE GEMM entrypoints below cover 16-bit, 8-bit, and mixed-dtype
W4A8 paths. ``GemmMixedDType.S4FP8`` uses
``b_quant_recipe=(1, 128)``. For ``masked_moe_gemm_mixed_dtype``, compatible
S4FP8 decode workloads can use ``backend="mutlass"``. ``backend="auto"``
selects MUTLASS for supported non-overlap inputs, generally when the dispatch
M is at most 32. Grouped-A ``a_quant_recipe=(1, 128)`` requires MUTLASS.
``GemmMixedDType.FP4FP8`` uses E2M1 weights, E8M0 residual scales, an FP32
epilogue scale, ``a_quant_recipe=(1, -1)``, and
``b_quant_recipe=(1, 32)`` with the MUBIN backend.

.. autofunction:: ragged_m_moe_gemm_8bit
.. autofunction:: ragged_m_moe_gemm_16bit
.. autofunction:: ragged_k_moe_gemm_8bit
.. autofunction:: ragged_k_moe_gemm_16bit
.. autofunction:: masked_moe_gemm_8bit
.. autofunction:: masked_moe_gemm_16bit
.. autofunction:: ragged_moe_gemm_mixed_dtype
.. autofunction:: masked_moe_gemm_mixed_dtype

Dense GEMM
-------------------------

.. currentmodule:: mate.gemm

.. autofunction:: bmm

DeepGemm Lighting Indexer
-------------------------

.. currentmodule:: mate.deep_gemm
.. autofunction:: fp8_einsum
.. autofunction:: tf32_hc_prenorm_gemm
.. autofunction:: fp8_mqa_logits
.. autofunction:: fp8_gemm_nt_skip_head_mid
.. autofunction:: get_paged_mqa_logits_metadata
.. autofunction:: fp8_paged_mqa_logits
