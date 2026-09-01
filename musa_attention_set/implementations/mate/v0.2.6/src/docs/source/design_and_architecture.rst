MATE Design and Architecture
============================

This document provides a technical overview of the MUSA AI Tensor Engine
(MATE). It explains MATE's layered architecture, its CUDA-ecosystem
compatibility approach, and its optimization focus for running generative AI
workloads efficiently on Moore Threads GPUs.

Architectural Layers
--------------------

MATE is a layered runtime and compilation engine that sits between
high-level inference frameworks and MUSA-native execution backends.

.. raw:: html

   <pre class="mermaid">
   flowchart TB
       frameworks["1. Framework Integration Layer<br/>vLLM, SGLang"]
       compat["2. CUDA Ecosystem Compatibility Layer<br/>FlashAttention-3, SageAttention, FlashMLA, MSA, FlashKDA, FlashInfer, DeepGEMM"]
       core["3. MATE Core<br/>mha_interface.py, flashmla.py, gemm.py, deep_gemm.py"]
       kernels["4. Native MUSA Kernels and Execution Artifacts<br/>MUBIN, TileLang, JIT, AOT, MUTLASS"]

       frameworks --> compat --> core --> kernels
   </pre>

1. Framework Integration Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MATE integrates with inference and serving frameworks through:

- wrapper-first integration surfaces when a matching wrapper is available
- direct ``mate`` APIs for advanced usage or when wrapper coverage does not
  match a workload

Design intent:

- keep integration familiar for users coming from CUDA-oriented ecosystems
- reduce the cost of bringing existing inference stacks onto MUSA hardware
- provide a clear path to adopt MATE capabilities incrementally

2. CUDA Ecosystem Compatibility Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This layer provides compatibility for selected CUDA-oriented operator
interfaces by mapping them onto MUSA-backed execution paths.

- FlashAttention-3
- SageAttention
- FlashMLA
- FlashKDA
- DeepGEMM

.. note::

   Compatibility is interface-oriented and scope-specific. Coverage can vary
   by operator family, tensor shape constraints, and version.

   When a wrapper is not available or not applicable, users can integrate
   through direct ``mate`` APIs.

3. MATE Core
~~~~~~~~~~~~

MATE Core is responsible for:

- public API entrypoints
- operator selection and dispatch
- scheduling and execution orchestration
- tensor layout handling and data movement conventions
- bridging wrapper calls to native MUSA kernel backends

4. Native MUSA Kernels and Compiled Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This layer provides the hardware-executed implementation of MATE operators.
MATE supports multiple backend building blocks and packaging forms, including:

- version-pinned MUBIN metadata and kernel artifacts
- TileLang kernels
- JIT-managed compiled modules
- AOT-packaged libraries
- MUTLASS-backed build inputs for selected compiled paths

MUBIN artifact delivery
^^^^^^^^^^^^^^^^^^^^^^^

The ``gemm``, ``flash_attention``, ``flash_mla``, and ``sage_attention``
families have MUBIN-backed paths whose generated launch artifacts are kept
outside the main ``mate`` wheel. The Python validation, dispatch, and JIT launch
code ships with ``mate``. External artifacts contain ``kernel_map.json`` and
``.o`` kernel objects.

MATE resolves these artifacts in the following order:

1. An installed ``mate-mubin`` package.
2. The downloaded cache selected by ``MATE_MUBIN_DIR``.
3. The default ``~/.cache/mate/mubin`` downloaded cache.

The packaged form is trusted to contain the complete payload. In cache-backed
operation, MATE first retrieves and verifies ``kernel_map.json`` for the
selected module. Local MATE dispatch code retrieves the selected ``.o`` kernel
lazily. Downloads use per-destination file locking, temporary files, and atomic
replacement so concurrent processes do not expose partial artifacts.

Artifact paths and kernel-map hashes are pinned by MATE. Hash verification is
enabled by default for downloaded kernel maps and kernel objects.

MUBIN artifact status is separate from the registered JIT/AOT module registry.
Use ``mate list-mubins`` for MUBIN status; ``mate module-status`` only describes
registered JIT and AOT modules.

Model and Workload Coverage
---------------------------

MATE focuses optimization on these core generative AI operator families:

- Attention and FMHA
- GEMM
- MLA
- KDA
- GDN
- Hyperconnection
- MoE-adjacent routing-related paths

MATE also includes model-tuned paths for selected mainstream model families,
depending on version and configuration, including:

- DeepSeek V3, V3.1, and V3.2 MLA-related paths
- DeepSeek V4 hyperconnection and routing-related paths
- Qwen-related GDN configurations

For detailed constraints and supported configurations, see the relevant
support matrices and compatibility pages. For GDN, see :doc:`gdn`.

In workload terms, MATE most clearly supports:

- Text generation paths built around prefill and decode
- Inference-serving integrations through wrapper-first package surfaces
- Long-context and KV-cache execution paths

Design Choices and Value
------------------------

MATE balances migration speed and performance through these design choices:

- Wrapper-first migration path. Reuse familiar CUDA-oriented operator surfaces
  where supported.
- Direct API fallback. Integrate through MATE APIs when wrapper coverage is
  insufficient.
- Hybrid execution approach. Support versioned MUBIN artifacts, AOT packaging,
  and runtime JIT compilation.
- Native MUSA optimization focus. Prioritize performance-critical operator
  families such as attention, GEMM, MLA, and related components.
- Operational readiness. Provide built-in diagnostics and reproducibility
  mechanisms, including logging, environment inspection, and dump-and-replay
  workflows for faster issue triage.

Execution Modes (MUBIN, AOT, and JIT)
-------------------------------------

MATE supports multiple deployment and execution modes:

- MUBIN artifacts. Selected operator families use version-pinned metadata and
  kernel artifacts supplied by ``mate-mubin`` or a verified download cache.
- AOT (Ahead-of-Time). Selected operator variants can be built and packaged
  before runtime for more predictable deployment behavior.
- JIT (Just-in-Time). Missing variants can be compiled on demand at runtime
  to improve coverage and flexibility.
- Preference and fallback behavior. When a matching AOT artifact is available,
  MATE can prefer AOT variants, while JIT behavior can be controlled through
  configuration and environment settings. This JIT/AOT fallback is separate
  from MUBIN package/cache resolution.

Operational System Tools
------------------------

When an operator crash, environment mismatch, or output issue needs deeper
investigation, MATE provides these built-in tools:

- ``mate check`` for runtime validation
- ``mate show-config`` for versions, devices, architecture resolution, and
  JIT/AOT status
- ``mate env`` for current MATE-related environment variables
- ``mate list-mubins`` for the active MUBIN source and per-module status
- API logging for call metadata, structured logs, and Level 10 dumps
- ``mate replay`` for deterministic replay of captured dumps

See Also
--------

- :doc:`Overview <overview>` for onboarding and getting started
- :doc:`Wrappers <wrapper_tutorials>` for the wrapper-first integration path
- :doc:`Python APIs <api_reference>` for direct integration entrypoints
- :doc:`Diagnostic Overview <diagnostics>` for configuration inspection and troubleshooting
- :doc:`Logging <logging_debugging>` for debugging workflows
