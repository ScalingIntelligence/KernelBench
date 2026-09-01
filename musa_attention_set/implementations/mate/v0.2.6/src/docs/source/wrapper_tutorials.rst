Wrappers
========

MATE uses the packages in the ``wrappers/`` directory as a compatibility
layer to run CUDA software on MUSA. They preserve familiar package names and
high-level APIs while routing execution to MATE operators and kernels. This
enables existing integrations to migrate to Moore Threads platforms with
minimal code changes.

Wrappers are the default integration path when your framework already targets
a supported CUDA-oriented Python package. For delivered packages, install the
matching wrapper first. ``pip`` installs the matching ``mate`` dependency
automatically from the same wheel source. Build from ``wrappers/`` only when
you are developing a wrapper locally.

How the wrappers work
---------------------

Each wrapper keeps the upstream-facing Python package surface stable while
routing supported execution paths to MATE-backed implementations on MUSA.

Key mechanisms
~~~~~~~~~~~~~~

- API mapping: Maps upstream-style calls to MATE operator paths.
- Namespace preservation: Preserves expected package names and import paths.
- Kernel routing: Runs calls on MATE-optimized operators and MUSA kernels.
- Distribution identification: Uses the PEP 440 local version suffix
  ``+musa`` so installed MUSA wrappers can be distinguished from native
  implementations with ``python -m pip show <package>``.

Why use wrappers
~~~~~~~~~~~~~~~~

- Lower migration overhead: Minimizes code changes and avoids separate
  hardware-specific code paths.
- Faster integration: Accelerates deployment of common tools and libraries on
  Moore Threads GPUs.

Wrapper support at a glance
---------------------------

Select a wrapper package to open its documentation page.

All wrapper packages in the table below are available from the MUSA Python
wheel source. Install the package that matches the import surface expected by
your framework.

MSA (MiniMax Sparse Attention) workloads map to the ``fmha_sm100`` wrapper.
Use it when your project already targets the ``fmha_sm100`` package surface
and you want the same import path on MUSA.

KDA (Kimi Delta Attention) workloads map to the ``flash_kda`` wrapper. Use it
when your project already targets the ``flash_kda`` package surface and you
want the same import path on MUSA.

.. list-table::
   :header-rows: 1

   * - Wrapper package
     - Import path
     - Best fit
     - Current scope
   * - :doc:`flash_attn_3 <wrappers/flash_attention_wrapper>`
     - ``flash_attn_interface``
     - FlashAttention-3 style APIs
     - Dense FMHA, varlen FMHA, KV-cache attention, scheduler metadata
   * - :doc:`sageattention <wrappers/sageattention_wrapper>`
     - ``sageattention``
     - SageAttention style APIs
     - Dense SageAttention-compatible path
   * - :doc:`flash_mla <wrappers/flash_mla_wrapper>`
     - ``flash_mla``
     - FlashMLA style APIs
     - MLA metadata, decode, sparse prefill
   * - :doc:`fmha_sm100 <wrappers/fmha_sm100_wrapper>`
     - ``fmha_sm100``
     - MSA style APIs
     - MSA planning, sparse prefill/decode, sparse top-k selection
   * - :doc:`flash_kda <wrappers/flash_kda_wrapper>`
     - ``flash_kda``
     - FlashKDA / KDA style APIs
     - KDA forward, workspace-size compatibility helper
   * - :doc:`flashinfer-python <wrappers/flashinfer_wrapper>`
     - ``flashinfer``
     - FlashInfer style APIs
     - BF16/FP8 GEMM, FP8 MLA RoPE quantization, and sparse decode
   * - :doc:`deep-gemm <wrappers/deep_gemm_wrapper>`
     - ``deep_gemm``
     - DeepGEMM style APIs
     - Grouped GEMM, dense GEMM, prenorm GEMM, MQA logits
