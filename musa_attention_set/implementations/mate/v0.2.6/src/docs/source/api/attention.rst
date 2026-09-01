.. _apiattention:

Attention
=========

Use the ``flash_attn_3`` or ``flash_mla`` wrapper packages first when your
integration already targets FlashAttention-3 or FlashMLA Python APIs. Use the
MATE APIs below when wrapper coverage is not enough.

FMHA
----

.. currentmodule:: mate.mha_interface

Forward dtype notes:

- ``flash_attn_varlen_func`` and ``flash_attn_with_kvcache`` accept
  ``torch.float16``, ``torch.bfloat16``, ``torch.float8_e4m3fn``, and
  ``torch.float8_e5m2`` inputs on the MATE FMHA forward path.
- For FP8 inputs, optional ``q_descale``, ``k_descale``, and ``v_descale``
  tensors are ``torch.float32`` scale factors with shape
  ``(batch_size, num_heads_kv)``.
- FP8 inputs produce ``torch.bfloat16`` outputs by default.
- The FMHA forward path also supports an optional ``qv`` input, including FP8
  inputs with ``qv``.
- When both ``q`` and ``qv`` are FP8, ``q_descale`` applies to both query
  tensors.
- ``flash_attn_with_kvcache(..., only_qv=True)`` skips the QK score path.
  In this mode, ``qv`` is required and ``q`` / ``k_cache`` may be ``None``.
- For best FP8 attention performance, use MUSA SDK 5.2.0 or newer when
  available.
- Use ``flash_attn_combine`` to merge partial FMHA outputs and log-sum-exp
  buffers when your integration splits the attention computation.
- Use ``get_scheduler_metadata`` to precompute the ``scheduler_metadata`` input
  for ``flash_attn_with_kvcache`` when you call the FMHA path directly.

.. autofunction:: flash_attn_combine
.. autofunction:: flash_attn_with_kvcache
.. autofunction:: get_scheduler_metadata
.. autofunction:: flash_attn_varlen_func


MLA
---

.. currentmodule:: mate.flashmla

FlashMLA supports multiple Q/K head-ratio configurations.

Minimal dense MLA example:

.. code-block:: python

   import torch
   from mate.flashmla import get_mla_metadata, flash_mla_with_kvcache

   q = torch.randn((1, 32, 8, 576), device="musa", dtype=torch.bfloat16)
   k_cache = torch.randn((4, 16, 2, 576), device="musa", dtype=torch.bfloat16)
   block_table = torch.zeros((1, 4), device="musa", dtype=torch.int32)
   cache_seqlens = torch.tensor([32], device="musa", dtype=torch.int32)

   tile_scheduler_metadata, num_splits = get_mla_metadata(
       cache_seqlens=cache_seqlens,
       num_q_tokens_per_head_k=q.shape[1] * q.shape[2] // k_cache.shape[2],
       num_heads_k=k_cache.shape[2],
       num_heads_q=q.shape[2],
       q=q,
       bs=q.shape[0],
   )

   out, lse = flash_mla_with_kvcache(
       q=q,
       k_cache=k_cache,
       block_table=block_table,
       cache_seqlens=cache_seqlens,
       head_dim_v=512,
       tile_scheduler_metadata=tile_scheduler_metadata,
       num_splits=num_splits,
   )

.. autofunction:: get_mla_metadata
.. autofunction:: flash_mla_with_kvcache
.. autofunction:: flash_mla_sparse_fwd
.. autofunction:: flash_mla_sparse_fwd_pack8
