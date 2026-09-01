Python APIs
===========

This section covers the direct MATE Python APIs.
Use these interfaces only when your framework or model architecture requires
direct symbol-level integration.

.. note::

   Check wrapper availability first. The wrapper-first workflow preserves more
   of the upstream package behavior and usually requires less code change.

Use direct MATE Python APIs when:

- no wrapper matches your framework's package surface
- a wrapper exists but does not cover the feature you need
- you need symbol-level control over a specific operator path

Supported API Entrypoints
-------------------------

FlashInfer wrapper
------------------

The :doc:`FlashInfer wrapper API reference <api/flashinfer>` documents the
supported ``flashinfer.rope`` and ``flashinfer.decode`` functions, including
their signatures, parameters, and return values.

Attention
---------

Optimized entrypoints for FlashAttention, varlen, KV-cache, scheduler
metadata, and MLA-related :doc:`attention <api/attention>` paths.

- ``mate.flash_attn_combine``
- ``mate.flash_attn_varlen_func``
- ``mate.flash_attn_with_kvcache``
- ``mate.get_scheduler_metadata``
- ``mate.get_mla_metadata``
- ``mate.flash_mla_with_kvcache``
- ``mate.flashmla.flash_mla_sparse_fwd``
- ``mate.flashmla.flash_mla_sparse_fwd_pack8``

Sparse MLA
----------

Native :doc:`sparse MLA <api/sparse_mla>` entrypoints for fused RoPE FP8
quantization, reusable decode scheduler metadata, and FP8 sparse decode.

- ``mate.sparse_mla_interface.mla_rope_quantize_fp8``
- ``mate.sparse_mla_interface.sparse_mla_fp8_decode``

SageAttention
-------------

Low-level pre-quantized SageAttention entrypoints for direct integration when
the ``sageattention`` wrapper is not the right surface.

- ``mate.sage_attn_quantized``
- ``mate.sage_attn_quantized_with_kvcache``

GEMM
----

Low-precision :doc:`GEMM <api/gemm>` entrypoints, including 8-bit and
mixed-dtype MoE GEMM, batched GEMM, and DeepGEMM-specific metadata / logits
helpers.

- ``mate.gemm.ragged_m_moe_gemm_8bit``
- ``mate.gemm.ragged_m_moe_gemm_16bit``
- ``mate.gemm.ragged_k_moe_gemm_8bit``
- ``mate.gemm.ragged_k_moe_gemm_16bit``
- ``mate.gemm.masked_moe_gemm_8bit``
- ``mate.gemm.masked_moe_gemm_16bit``
- ``mate.gemm.ragged_moe_gemm_mixed_dtype``
- ``mate.gemm.masked_moe_gemm_mixed_dtype``
- ``mate.gemm.bmm``
- ``mate.deep_gemm.fp8_einsum``
- ``mate.deep_gemm.fp8_mqa_logits``
- ``mate.deep_gemm.tf32_hc_prenorm_gemm``
- ``mate.deep_gemm.fp8_gemm_nt_skip_head_mid``
- ``mate.deep_gemm.get_paged_mqa_logits_metadata``
- ``mate.deep_gemm.fp8_paged_mqa_logits``

Mega MoE
--------

Distributed :doc:`Mega MoE <api/mega_moe>` entrypoints for DeepGEMM-style
expert execution across a process group. Use this path when you need the
fused distributed FP8 expert runtime rather than a single MoE GEMM operator.
The same APIs are available through ``mate.deep_gemm`` and ``mate.mega_moe``;
the ``mate.deep_gemm`` aliases are the usual integration surface.

- ``mate.deep_gemm.get_symm_buffer_for_mega_moe``
- ``mate.deep_gemm.transform_weights_for_mega_moe``
- ``mate.deep_gemm.fp8_fp8_mega_moe``

Hyperconnection
---------------

Direct :doc:`Hyperconnection <api/hyperconnection>` APIs exposed at the
top-level ``mate`` package and through ``mate.hyperconnection``.

- ``mate.mhc_pre``
- ``mate.mhc_prenorm_gemm_sqrsum``
- ``mate.mhc_pre_big_fuse``

GDN
---

Direct Gated Delta Network APIs for decode and prefill. See the
:doc:`GDN support page <gdn>` for support details and workflow guidance.

- ``mate.gated_delta_rule_decode``
- ``mate.gdn_prefill.chunk_gated_delta_rule``

KDA
---

Direct :doc:`KDA <api/kda>` entrypoints for fused chunked KDA and decode when
the ``flash_kda`` wrapper is not the right integration surface.

- ``mate.chunk_kda``
- ``mate.kda.chunk_kda``
- ``mate.kda.gated_delta_rule_decode``

MSA
---

MSA (MiniMax Sparse Attention) covers MATE's direct dense, paged, and sparse
attention APIs on MUSA. Use the ``fmha_sm100`` wrapper first when your project
already targets that package surface. Use the direct MATE APIs below when you
need the native planning and runtime contract.

- ``mate.msa_interface.msa_plan``
- ``mate.msa_interface.msa``
- ``mate.msa_interface.sparse_msa_plan``
- ``mate.msa_interface.sparse_msa``
- ``mate.msa_interface.sparse_topk_select``
- ``mate.msa_interface.sparse_decode_atten_func``

The detail page documents the plan types, runtime metadata, and page table
helper used by the direct API path.

MoE Routing & Gating
--------------------

Direct MoE routing and gating entrypoints for workflows that need MATE's
native router path instead of a wrapper-level integration.

- ``mate.hash_topk``
- ``mate.moe_fused_gate``
