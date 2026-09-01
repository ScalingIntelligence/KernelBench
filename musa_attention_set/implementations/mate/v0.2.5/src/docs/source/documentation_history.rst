Documentation History
=====================

This page records documentation changes by release.

For runtime, API, and compatibility changes, see
`GitHub Releases <https://github.com/MooreThreads/mate/releases>`_.

The current docs site history starts at ``0.2.2``.

0.2.5
-----

- Added Kimi Delta Attention (KDA) decode coverage to
  :doc:`Python APIs <api_reference>` and :doc:`KDA <api/kda>`.
- Expanded :doc:`Attention <api/attention>` for FMHA ``only_qv`` and MLA
  head-ratio coverage.
- Added the :doc:`MSA/fmha_sm100 wrapper <wrappers/fmha_sm100_wrapper>` for
  MiniMax Sparse Attention (MSA) workflows on MUSA.
- Updated :doc:`Installing MATE <install>` for the MUSA simple package index.
- Added MUBIN artifact management commands to :doc:`Command Line Interface
  <cli>`.
- Refined FlashMLA/DenseMLA support for multiple head-ratio configurations.

0.2.4
-----

- Updated :doc:`Installing MATE <install>` for wrapper-first ``pip``
  installation, including automatic ``mate`` dependency installation.
- Added W4A8 mixed-dtype MoE GEMM to :doc:`Python APIs <api_reference>` and
  :doc:`GEMM <api/gemm>`.
- Added Mega MoE to :doc:`Python APIs <api_reference>` and
  :doc:`Mega MoE <api/mega_moe>`.

0.2.3
-----

- Expanded the docs set around :doc:`Overview <overview>`,
  :doc:`Installing MATE <install>`, :doc:`Wrappers <wrapper_tutorials>`,
  :doc:`Diagnostic Overview <diagnostics>`, :doc:`Command Line Interface
  <cli>`, :doc:`Design and Architecture <design_and_architecture>`, and
  :doc:`Python APIs <api_reference>`.
- Added wrapper pages for :doc:`FlashAttention
  <wrappers/flash_attention_wrapper>`, :doc:`SageAttention
  <wrappers/sageattention_wrapper>`, :doc:`FlashMLA
  <wrappers/flash_mla_wrapper>`, :doc:`FlashKDA
  <wrappers/flash_kda_wrapper>`, and :doc:`DeepGEMM
  <wrappers/deep_gemm_wrapper>`.
- Added API pages for :doc:`Attention <api/attention>`,
  :doc:`GEMM <api/gemm>`, :doc:`HyperConnection <api/hyperconnection>`, and
  :doc:`KDA <api/kda>`.
- Moved :doc:`Command Line Interface <cli>`,
  :doc:`Environment Variables <environment_variables>`, and
  :doc:`GDN Support Matrix <gdn>` into the Sphinx site structure.

0.2.2
-----

- Established the initial public docs baseline in :doc:`the home page <index>`
  and the early API set.
- Added the :doc:`HyperConnection API page <api/hyperconnection>`.
- Updated :doc:`FlashAttention <api/attention>` and
  :doc:`GDN Support Matrix <gdn>`.
- Refreshed :doc:`the home page <index>` structure.
