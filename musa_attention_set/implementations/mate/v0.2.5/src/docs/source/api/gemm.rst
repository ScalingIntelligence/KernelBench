.. _apigemm:

GEMM
=========

For framework integrations that already target DeepGEMM style Python APIs,
prefer the ``deep-gemm`` wrapper first. Use the MATE APIs below when wrapper
coverage is not enough.

MoE GEMM
-------------------------

.. currentmodule:: mate.gemm

The direct MoE GEMM entrypoints below cover both the existing 8-bit paths and
mixed-dtype W4A8 paths. ``GemmMixedDType.S4FP8`` uses
``b_quant_recipe=(1, 128)``; ``GemmMixedDType.FP4FP8`` uses E2M1 weights,
E8M0 residual scales, an FP32 epilogue scale, and
``b_quant_recipe=(1, 32)``. Both use ``backend="mubin"`` and
``a_quant_recipe=(1, -1)``.

.. autofunction:: ragged_m_moe_gemm_8bit
.. autofunction:: ragged_k_moe_gemm_8bit
.. autofunction:: masked_moe_gemm_8bit
.. autofunction:: ragged_moe_gemm_mixed_dtype
.. autofunction:: masked_moe_gemm_mixed_dtype

Dense GEMM
-------------------------

.. autofunction:: bmm_fp16
.. autofunction:: bmm_fp8
.. autofunction:: gemm_fp8_nt_groupwise

DeepGemm Lighting Indexer
-------------------------

.. currentmodule:: mate.deep_gemm
.. autofunction:: get_paged_mqa_logits_metadata
.. autofunction:: fp8_paged_mqa_logits
