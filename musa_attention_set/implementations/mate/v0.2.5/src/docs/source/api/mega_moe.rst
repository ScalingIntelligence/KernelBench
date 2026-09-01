.. _apimega_moe:

Mega MoE
========

Mega MoE is MATE's distributed FP8 Mixture-of-Experts execution path for
multi-rank expert-parallel workloads. Use it when your integration already has
token activations, top-k expert indices, and top-k expert weights prepared and
you need MATE to run the expert compute path across a process group.

For integrations that already target DeepGEMM-style distributed MoE
execution, prefer the ``deep-gemm`` wrapper first. Use the direct MATE APIs
below when wrapper coverage is not enough.

The examples here use the ``mate.deep_gemm`` aliases. The same entrypoints are
also available from ``mate.mega_moe``.

Use Mega MoE when
-----------------

- you are integrating a distributed expert-parallel MoE path across multiple
  ranks
- your expert weights are stored as grouped FP8 tensors
- you need the fused two-layer expert execution path, not just one grouped
  GEMM call

Use the :doc:`GEMM APIs <gemm>` instead when you only need grouped or masked
expert GEMM operators, including the ``0.2.4`` W4A8 mixed-dtype GEMM path.

Typical Flow
------------

Mega MoE uses a shared runtime workspace that holds FP8 inputs, routing
metadata, and intermediate activations for the fused execution path.

1. Allocate the symmetric runtime buffer with
   ``get_symm_buffer_for_mega_moe``.
2. Transform logical FP8 expert weights with
   ``transform_weights_for_mega_moe``.
3. Launch the kernel with ``fp8_fp8_mega_moe``.

Current Limits
--------------

- Intranode process groups only, with up to ``8`` ranks.
- ``num_experts`` must be divisible by the process-group size.
- ``activation="swiglu"`` only.
- ``use_fp8_dispatch=True`` only.
- ``hidden`` must be divisible by ``128``.
- ``intermediate_hidden`` must be divisible by ``256``.
- ``fp8_fp8_mega_moe`` writes into a contiguous ``torch.bfloat16`` output
  tensor.

API
---

.. py:function:: mate.deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, use_fp8_dispatch=True, activation="swiglu")

   Allocate the shared runtime workspace used by Mega MoE execution.

.. py:function:: mate.deep_gemm.transform_weights_for_mega_moe(l1_weights, l2_weights=None)

   Transform grouped FP8 expert weights into the layout expected by the Mega
   MoE runtime.

.. py:function:: mate.deep_gemm.fp8_fp8_mega_moe(y, l1_weights, l2_weights, sym_buffer, cumulative_local_expert_recv_stats=None, recipe=(1, 1, 32), activation="swiglu", activation_clamp=None, fast_math=True)

   Run the FP8 Mega MoE expert compute path and write the result to ``y``.
