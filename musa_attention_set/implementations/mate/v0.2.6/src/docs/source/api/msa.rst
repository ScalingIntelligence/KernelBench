.. _apimsa:

MSA
===

.. currentmodule:: mate.msa_interface

Use the ``fmha_sm100`` wrapper first when your project already targets the MSA
package surface. Use the direct MATE APIs below when you need the native plan
and runtime contract.

MSA (MiniMax Sparse Attention) covers dense, paged, and sparse attention on
MUSA.

The ``mate`` package also exports the primary planning and execution
entrypoints at the top level.

Dense and paged paths
---------------------

.. autofunction:: msa_plan
.. autofunction:: msa

.. autoclass:: MsaPlan
.. autoclass:: MsaPrefillPlan
.. autoclass:: MsaDecodePlan
.. autoclass:: MsaRuntimeMetadata
.. autodata:: MsaPlanInfo

Use ``build_page_table_from_flat_kv_indices`` to turn flat page indices into a
fixed-shape page table for paged MSA.

Sparse paths
------------

.. autofunction:: sparse_msa_plan
.. autofunction:: sparse_msa
.. autofunction:: sparse_topk_select
.. autofunction:: sparse_decode_atten_func
