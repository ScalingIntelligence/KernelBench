.. _apikda:

KDA
===

.. currentmodule:: mate.kda

For framework integrations that already target FlashKDA Python APIs, prefer
the ``flash_kda`` wrapper package first. Use the MATE API below when wrapper
coverage is not enough or when direct operator-level control is required.

KDA (Kimi Delta Attention) covers MATE's direct operator APIs for chunked KDA
and KDA decode on MUSA. Use this page when the ``flash_kda`` wrapper is not
enough, or when you need direct control over ``mate.kda`` entrypoints such as
``chunk_kda`` and ``gated_delta_rule_decode``.

``chunk_kda`` is MATE's fused chunked KDA operator on MUSA, while
``gated_delta_rule_decode`` provides the direct KDA decode API.

Minimal KDA example:

.. code-block:: python

   import torch
   from mate.kda import chunk_kda

   q = torch.randn((1, 4, 2, 128), device="musa", dtype=torch.bfloat16)
   k = torch.randn((1, 4, 2, 128), device="musa", dtype=torch.bfloat16)
   v = torch.randn((1, 4, 4, 128), device="musa", dtype=torch.bfloat16)
   g = torch.randn((1, 4, 4, 128), device="musa", dtype=torch.bfloat16)
   beta = torch.randn((1, 4, 4), device="musa", dtype=torch.bfloat16)

   output = chunk_kda(q=q, k=k, v=v, g=g, beta=beta)

Chunk KDA at a glance
---------------------

- Public API: ``mate.chunk_kda`` / ``mate.kda.chunk_kda``
- Device: MUSA
- Input dtypes: ``torch.float16`` and ``torch.bfloat16``
- Head dimension: currently fixed to ``128``
- Sequence modes:
  - Dense: ``[B, T, H, 128]``
  - Varlen: ``[S, H, 128]`` or ``[1, S, H, 128]`` with ``cu_seqlens``
- Optional recurrent state input/output
- Optional preallocated ``output`` and ``final_state``

Chunk KDA toolchain requirements
--------------------------------

- The repository-wide install baseline still applies, but build the current
  fused chunk KDA path with MUSA SDK / MTCC 5.1.0 or newer.
- The 4.3.6 toolchain may fail to compile KDA kernels.

Chunk KDA shape contract
------------------------

For dense mode:

- ``q`` / ``k``: ``[B, T, Hqk, 128]``
- ``v`` / ``g``: ``[B, T, Hv, 128]``
- ``beta``: ``[B, T, Hv]``

For varlen mode:

- ``q`` / ``k``: ``[S, Hqk, 128]`` or ``[1, S, Hqk, 128]``
- ``v`` / ``g``: ``[S, Hv, 128]`` or ``[1, S, Hv, 128]``
- ``beta``: ``[S, Hv]`` or ``[1, S, Hv]``
- ``cu_seqlens``: cumulative sequence lengths with shape ``[num_seqs + 1]``

Additional constraints:

- ``k.shape == q.shape``
- ``g.shape == v.shape``
- ``beta.shape == v.shape[:3]``
- ``Hv`` must be divisible by ``Hqk``
- the last dimension of every tensor passed to the kernel must be contiguous

Chunk KDA state tensors
-----------------------

- ``initial_state`` is optional
- ``final_state`` is optional unless ``output_final_state=True``
- state shape: ``[num_seqs, Hv, 128, 128]``
- state dtype: same as value dtype or ``torch.float32``
- if both ``initial_state`` and ``final_state`` are provided, their dtypes must match

Chunk KDA gate parameters
-------------------------

When gate parameters are enabled:

- ``A_log`` and ``dt_bias`` must be provided together
- ``A_log`` shape: ``[Hv]``
- ``dt_bias`` shape: ``[Hv, 128]``
- ``lower_bound`` defaults to ``-5.0``

Chunk KDA outputs
-----------------

- default return: ``output``
- if ``output_final_state=True``: returns ``(output, final_state)``

Decode support
--------------

The decode entry point is ``mate.kda.gated_delta_rule_decode``. It remains
module-scoped because ``mate.gated_delta_rule_decode`` is the separate GDN API.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Capability
     - Supported KDA decode scope
   * - Device
     - MUSA through the TileLang backend
   * - Sequence layout
     - Dense ``[B, T, H, D]`` or varlen ``[1, total_tokens, H, D]`` with
       ``cu_seqlens``
   * - Q/K/V dtypes
     - Matching ``torch.float16`` or ``torch.bfloat16`` tensors
   * - Head dimensions
     - ``K == V`` and the dimension must be divisible by ``32``
   * - Head grouping
     - The number of value/state heads must be divisible by the number of Q/K
       heads
   * - State
     - Optional ``torch.float32`` or ``torch.bfloat16`` state in V-first
       ``[pool, HV, V, K]`` or K-first ``[pool, HV, K, V]`` layout
   * - Output dtype
     - ``torch.float16``, ``torch.bfloat16``, or ``torch.float32``
   * - Serving modes
     - Fixed or variable-length decode, continuous batching through
       ``state_indices``, and speculative decode through
       ``num_accepted_tokens``
   * - State output
     - Separate final-state output or in-place state update

Decode inputs use ``a`` with shape ``[B, T, HV, K]`` and ``b`` with shape
``[B, T, HV]`` or ``[B, T, HV, V]``. ``A_log`` is optional float32 metadata
with shape ``[HV]``; ``dt_bias`` is optional float32 or bfloat16 data with
shape ``[HV, K]``.

API reference
-------------

.. autofunction:: chunk_kda

.. autofunction:: gated_delta_rule_decode
