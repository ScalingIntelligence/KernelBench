Logging
=======

MATE logging helps you trace API calls, inspect inputs and outputs, and capture
replayable dumps for crash reproduction. At Level 10, MATE acts as a flight
recorder: it saves inputs before execution, appends outputs after successful
execution, and writes JSONL metadata for scanning and replay.

Quick Start
-----------

Enable basic logging:

.. code-block:: bash

   export MATE_LOGLEVEL=3
   export MATE_LOGDEST=stdout

Enable replayable dumps:

.. code-block:: bash

   export MATE_LOGLEVEL=10
   export MATE_DUMP_DIR=mate_dumps

Logging Levels
--------------

.. list-table::
   :header-rows: 1

   * - Level
     - Name
     - What it records
     - Best for
   * - ``0``
     - Disabled
     - No logging. The decorator returns the original function unchanged.
     - Production
   * - ``1``
     - Function Names
     - Function names only, logged before execution.
     - Basic tracing
   * - ``3``
     - Inputs and Outputs
     - Function names, arguments, outputs, and structured tensor metadata.
     - Standard debugging
   * - ``5``
     - Statistics
     - Level 3 plus tensor statistics such as min, max, mean, and NaN or Inf counts.
     - Numerical debugging
   * - ``10``
     - Flight Recorder
     - Level 5 plus on-disk tensor dumps and replay metadata.
     - Crash reproduction and replay

Environment Variables
---------------------

Main Configuration
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Variable
     - Type
     - Default
     - Description
   * - ``MATE_LOGLEVEL``
     - ``int``
     - ``0``
     - Logging level. Supported values are ``0``, ``1``, ``3``, ``5``, and ``10``.
   * - ``MATE_LOGDEST``
     - ``str``
     - ``stdout``
     - Log destination: ``stdout``, ``stderr``, or a file path. Use ``%i`` in a file path to inject the current process ID.

Dump Configuration (Level 10)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``MATE_LOGLEVEL=10``, the following variables control dump behavior:

.. list-table::
   :header-rows: 1

   * - Variable
     - Type
     - Default
     - Description
   * - ``MATE_DUMP_DIR``
     - ``str``
     - ``mate_dumps``
     - Root directory for dump files.
   * - ``MATE_DUMP_MAX_SIZE_GB``
     - ``float``
     - ``20``
     - Maximum total dump size per process, in gigabytes.
   * - ``MATE_DUMP_MAX_COUNT``
     - ``int``
     - ``1000``
     - Maximum number of API calls to dump per process.
   * - ``MATE_DUMP_INCLUDE``
     - ``str``
     - empty
     - Comma-separated include patterns in ``fnmatch`` style. Applied first.
   * - ``MATE_DUMP_EXCLUDE``
     - ``str``
     - empty
     - Comma-separated exclude patterns in ``fnmatch`` style. Applied after include filtering.
   * - ``MATE_DUMP_SAFETENSORS``
     - ``int``
     - ``0``
     - Set to ``1`` to write ``.safetensors`` files instead of ``.pt`` files.

Level 10 Dumping and Replay
---------------------------

Level 10 is MATE's replayable dump mode. When it is enabled, MATE:

1. creates a per-call dump directory
2. saves input tensors before execution
3. writes a metadata record with ``execution_status: "inputs_saved"``
4. executes the function
5. saves output tensors after successful execution
6. appends a second metadata record with ``execution_status: "completed"``

This design makes dumps crash-safe. If a process fails after inputs are saved
but before outputs are written, the input dump and the first metadata record are
still available for inspection or replay.

Typical workflow:

.. code-block:: bash

   export MATE_LOGLEVEL=10
   export MATE_DUMP_DIR=mate_dumps
   python app.py
   mate list-dumps mate_dumps/
   mate replay --dir mate_dumps/

``mate replay`` accepts either the dump root directory or a single dump
subdirectory.

Dump Filtering
--------------

Use ``MATE_DUMP_INCLUDE`` and ``MATE_DUMP_EXCLUDE`` to control which API calls
are written to disk.

Pattern syntax:

- ``*`` matches any number of characters
- ``?`` matches a single character
- matching is case-sensitive
- method names are recorded as ``ClassName.method_name`` when applicable

Filter logic:

- if ``MATE_DUMP_INCLUDE`` is set, only matching APIs are dumped
- if ``MATE_DUMP_EXCLUDE`` is set, matching APIs are skipped
- include filtering runs first, then exclude filtering

Examples:

.. code-block:: bash

   export MATE_DUMP_INCLUDE="*attention*,*gemm*"
   export MATE_DUMP_EXCLUDE="*.__init__,*.plan"

SafeTensors Format
------------------

By default, MATE writes dump tensors with ``torch.save``. This preserves stride
and non-contiguous layout information.

To use ``safetensors`` instead:

.. code-block:: bash

   export MATE_DUMP_SAFETENSORS=1

.. warning::

   ``safetensors`` does not preserve tensor strides or non-contiguous layout.
   Tensors are saved as contiguous. Use the default ``torch.save`` format when
   stride preservation matters for debugging.

Replay is format-aware. MATE automatically loads ``inputs.pt`` or
``inputs.safetensors``, and ``outputs.pt`` or ``outputs.safetensors``, based on
which files exist in the dump directory.

Dump Directory Structure
------------------------

When Level 10 logging is enabled, MATE writes a root session log and one
subdirectory per dumped API call.

.. code-block:: text

   MATE_DUMP_DIR/
   ├── session.jsonl
   ├── 20260601_120000_123_pid12345_<function_name>_call0001/
   │   ├── metadata.jsonl
   │   ├── inputs.pt              # or inputs.safetensors
   │   └── outputs.pt             # or outputs.safetensors
   └── ...

How to read this structure:

- ``session.jsonl`` is the session-wide event log. Each record is one JSON line.
- ``metadata.jsonl`` is the per-dump record file.
- the first record uses ``execution_status: "inputs_saved"``
- the second record uses ``execution_status: "completed"``
- if outputs are missing, the process may have failed after input capture

Process ID Substitution
-----------------------

Use ``%i`` in ``MATE_LOGDEST`` file paths for automatic process ID substitution.
This is useful for multi-process or multi-GPU jobs.

.. code-block:: bash

   export MATE_LOGLEVEL=3
   export MATE_LOGDEST="logs/mate_api_%i.log"

This produces per-process log files such as ``logs/mate_api_12345.log``.

Related Diagnostics Variables
-----------------------------

These variables are not logging controls, but they are often used together with
logging during offline debugging or build investigation.

.. list-table::
   :header-rows: 1

   * - Variable
     - Typical value
     - Purpose
   * - ``MATE_MUSA_ARCH_LIST``
     - ``3.1``
     - Set the architecture explicitly for offline diagnostics or build workflows.
   * - ``MATE_DISABLE_JIT``
     - ``1``
     - Disable runtime JIT and require matching AOT modules.
   * - ``MATE_JIT_VERBOSE``
     - ``1``
     - Print verbose runtime JIT build output.

For the full MATE environment variable reference, see
:doc:`Environment Variables <environment_variables>`.

Advanced Notes
--------------

- At Level 5, tensor statistics are skipped during MUSA graph capture to avoid synchronization issues.
- At Level 0, the decorator returns the original function unchanged, so logging has zero overhead.
- Replay can load a single dump directory or replay a sequence from the dump root.

See Also
--------

- For CLI commands such as ``mate replay`` and ``mate list-dumps``, see :doc:`cli`.
- For the broader troubleshooting path, see :doc:`diagnostics`.
