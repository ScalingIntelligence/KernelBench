Diagnostic Overview
===================

Use this section when installation, wrapper integration, or direct ``mate``
Python API usage does not behave as expected.

Quick Start
-----------

1. If you installed a compatibility wrapper, run
   ``python -m pip show <package>`` and confirm that its version ends in
   ``+musa``. This identifies the MATE-backed MUSA wrapper rather than the
   native implementation.
2. Run ``mate check`` to validate the runtime environment.
3. Run ``mate show-config`` to inspect versions, devices, architecture
   resolution, and JIT or AOT state.
4. Run ``mate env`` to confirm the shell exports seen by MATE.
5. Enable logging or Level 10 dumps when the failure requires deeper evidence.
6. Replay or share the captured dump data when the issue must be reproduced.

.. note::

   If one of the initial checks already explains the issue, you can stop there.

Primary Commands
----------------

.. code-block:: bash

   mate check
   mate show-config
   mate env

Next Steps
----------

.. list-table::
   :header-rows: 1

   * - Topic
     - Use Case
   * - :doc:`Command Line Interface <cli>`
     - Runtime checks, environment inspection, module status, or dump replay.
   * - :doc:`Logging <logging_debugging>`
     - Log levels, dump controls, environment variables, or replayable captures.

See Also
--------

- Go back to :doc:`Overview <overview>` if you are still choosing an integration path.
- Go back to :doc:`Wrappers <wrapper_tutorials>` if you are still working through a wrapper quickstart.
- Continue with :doc:`API Reference <api_reference>` if the failure is in direct ``mate`` Python API usage.
