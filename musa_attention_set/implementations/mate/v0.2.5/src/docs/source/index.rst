Welcome to MATE Documentation
=============================

`GitHub <https://github.com/MooreThreads/mate>`_

MATE (**M**\USA **A**\I **T**\ensor **E**\ngine) is a high-performance
operator library optimized for generative AI workloads on Moore Threads GPUs.
Built on TVM-FFI, it delivers highly efficient Transformer and large language
model (LLM) operator implementations, including Attention and GEMM.

MATE is specifically designed for framework developers and integration
engineers aiming to port existing CUDA-oriented pipelines to Moore Threads
GPUs with minimal code rewriting. To ease this transition, it features
CUDA-compatible Python wrappers that preserve familiar package surfaces. When a
compatible wrapper is unavailable, developers can directly use native
``mate`` APIs.

To streamline deployment and troubleshooting, the library supports a
wrapper-first integration path backed by robust runtime checks,
comprehensive logging, and environment inspection tools.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Overview <overview>
   Installing MATE <install>
   Release Notes <https://github.com/MooreThreads/mate/releases>

.. toctree::
   :maxdepth: 2
   :titlesonly:
   :caption: Wrapper Quickstarts

   Wrappers <wrapper_tutorials>
   FlashAttention Wrapper <wrappers/flash_attention_wrapper>
   SageAttention Wrapper <wrappers/sageattention_wrapper>
   FlashMLA Wrapper <wrappers/flash_mla_wrapper>
   MSA Wrapper <wrappers/fmha_sm100_wrapper>
   FlashKDA Wrapper <wrappers/flash_kda_wrapper>
   DeepGEMM Wrapper <wrappers/deep_gemm_wrapper>

.. toctree::
   :maxdepth: 1
   :caption: Deep Dive

   Design and Architecture <design_and_architecture>
   Choosing Wrappers vs. Python APIs <choosing_wrappers_vs_python_apis>

.. toctree::
   :maxdepth: 1
   :caption: Support & Compatibility

   GDN Support Matrix <gdn>
   FlashAttention3 Forward Compatibility <wrappers/flash_attention_forward_compatibility>

.. toctree::
   :maxdepth: 1
   :titlesonly:
   :caption: CLI & Diagnostics

   Diagnostic Overview <diagnostics>
   Command Line Interface <cli>
   Logging <logging_debugging>
   Environment Variables <environment_variables>
   Guard Allocator Debugging <guard_allocator>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   Python APIs <api_reference>
   Attention <api/attention>
   GEMM <api/gemm>
   Mega MoE <api/mega_moe>
   HyperConnection <api/hyperconnection>
   KDA <api/kda>

.. toctree::
   :maxdepth: 1
   :caption: Appendix

   Documentation History <documentation_history>
