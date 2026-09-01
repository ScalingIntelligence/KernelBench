Choosing Wrappers vs. Python APIs
=================================

Use this page to decide whether to integrate MATE through a
**compatibility wrapper** or through **direct MATE Python APIs**.

Default Recommendation
----------------------

**Start with a wrapper** when one matches your framework's existing
CUDA-oriented package surface, including imports and high-level API shape.
This typically minimizes migration work.

Use **direct MATE Python APIs** when:

- no wrapper matches your framework or package surface
- the wrapper does not cover a required feature or operator variant
- you need more explicit control over operator-level entrypoints

Decision Guide
--------------

.. list-table::
   :header-rows: 1

   * - If you need...
     - Prefer...
     - Why
   * - Minimal code change for an existing CUDA-oriented package surface
     - :doc:`Wrappers <wrapper_tutorials>`
     - Wrappers preserve familiar package names and route supported execution
       paths to MATE on MUSA.
   * - A supported FlashAttention-3, SageAttention, FlashMLA, FlashKDA, or DeepGEMM-style surface
     - :doc:`Wrappers <wrapper_tutorials>`
     - Wrapper pages define the supported scope and provide quickstart steps.
   * - Direct control over callable MATE entrypoints
     - :doc:`Python APIs <api_reference>`
     - Direct APIs provide explicit operator entrypoints and configuration
       control.
   * - A feature that is not exposed through a wrapper
     - :doc:`Python APIs <api_reference>`
     - Direct APIs let you integrate the required operator path without
       waiting on wrapper expansion.

Tradeoffs at a Glance
---------------------

.. list-table::
   :header-rows: 1

   * - Aspect
     - Wrappers
     - Python APIs
   * - Integration effort
     - Lower, when a matching wrapper exists
     - Higher, because you wire MATE entrypoints into your code
   * - Upstream package compatibility
     - Higher
     - Lower
   * - Operator-level control
     - Narrower
     - Higher
   * - Recommended starting point
     - Default path
     - Fallback / advanced

Examples
--------

- **Wrapper-first example:** Your framework already imports a supported surface
  such as FlashAttention-3. Install the matching wrapper, let it pull in the
  matching ``mate`` dependency automatically, keep the import path stable, then
  validate with ``mate check``.
- **Direct API example:** You need a specific operator entrypoint or variant
  not covered by wrappers. Call the MATE Python API directly for that operator
  family and integrate it into your application or module.

Next Steps
----------

- If a wrapper matches your stack, start with :doc:`Wrappers <wrapper_tutorials>`
  and the relevant wrapper quickstart.
- If you need direct entrypoints, go to :doc:`Python APIs <api_reference>` and
  the operator family reference for :doc:`Attention <api/attention>`,
  :doc:`GEMM <api/gemm>`, :doc:`Mega MoE <api/mega_moe>`,
  :doc:`HyperConnection <api/hyperconnection>`, or :doc:`KDA <api/kda>`.
- If runtime validation fails, start with
  :doc:`CLI & Diagnostics <diagnostics>` using ``mate check``,
  ``mate show-config``, and ``mate env``.
