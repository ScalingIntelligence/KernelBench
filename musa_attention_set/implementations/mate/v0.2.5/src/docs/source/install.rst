Installing MATE
===============

This page describes how to install MATE or a supported wrapper package from
the MUSA Python wheel source, or how to build MATE from source on top of an
existing MUSA-enabled ``torch`` / ``torch_musa`` stack.

Steps at a glance
-----------------

1. Check requirements.
2. Choose a package source.
3. Install a delivered package or build from source.
4. Validate installation and MUBIN artifact availability.
5. Build a wrapper from source if needed.
6. Optionally pre-build AOT kernels.

Step 1. Check Requirements
--------------------------

MATE currently requires the following runtime baseline:

+----------------+----------------------+
| Component      | Requirement          |
+================+======================+
| Python         | ``3.10`` or later    |
+----------------+----------------------+
| MUSA Toolkit   | ``4.3.6`` or later   |
+----------------+----------------------+
| TorchMUSA      | ``2.7`` or later     |
+----------------+----------------------+
| Architecture   | ``Pinghu (MP31)``    |
+----------------+----------------------+

The table above shows the repository-wide baseline. Some feature paths need
newer toolchains. The current external delivery source mainly covers
``x86_64`` and Python ``3.10`` / ``3.12`` wheels. When a wrapper or API page
lists a stricter requirement, follow that page. For example, FlashKDA
currently builds on MUSA SDK / MTCC 5.1.0+, and FlashAttention ``Local +
attention_chunk`` requires MUSA SDK 5.1.0+.

Before continuing, make sure the MUSA-enabled ``torch`` / ``torch_musa`` stack
is already installed and working in your environment.

Step 2. Choose a Package Source
-------------------------------

For delivered packages, use the external MUSA wheel source.

- Index:
  ``https://dl.mthreads.com/repo/api/pypi/pypi/simple``

Choose one configuration method:

Temporary shell or CI
   .. code-block:: bash

      export PIP_INDEX_URL=https://dl.mthreads.com/repo/api/pypi/pypi/simple

Persistent pip config
   .. code-block:: bash

      python -m pip config set global.index-url \
        https://dl.mthreads.com/repo/api/pypi/pypi/simple

One-off install
   .. code-block:: bash

      python -m pip install <package> \
        --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   .. important::

      Use one package index per install step. Do not mix the MUSA wheel
      source with a public PyPI mirror in the same ``pip install`` command.

Optional:

Pinned version install
   Use this when you need one exact version.

   .. code-block:: bash

      python -m pip install \
        <package>==<version> \
        --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   Keep one ``index-url`` configured for dependency resolution, or add
      ``--no-deps`` if dependencies are already installed.

Check available versions
   .. code-block:: bash

      python -m pip index versions mate \
        --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

   If ``pip index versions`` is unavailable, upgrade ``pip`` first.

Step 3. Install a Delivered Package or Build from Source
--------------------------------------------------------

Choose one installation path:

- Delivered wrapper install: recommended when your framework already expects
  ``flash_attn_3``, ``flash_mla``, ``flash_kda``, ``deep-gemm``, or
  ``sageattention``. Each delivered wrapper installs the matching ``mate``
  dependency automatically.
- Local wrapper install: use this for wrapper surfaces that are not part of
  the delivered package set, such as ``fmha_sm100``.
- Direct MATE install: use this when you need direct ``mate`` APIs without a
  wrapper.
- Build from source: use this when you are developing MATE locally or need a
  local build artifact.

MUBIN package options
~~~~~~~~~~~~~~~~~~~~~

MATE provides two packages for MUBIN-backed workloads:

- ``mate``: core package that compiles or downloads MUBIN kernels on first use.
- ``mate-mubin``: prebuilt kernels and runtime artifacts for faster startup and
  offline use.

To install both packages:

.. code-block:: bash

   python -m pip install mate mate-mubin \
     --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

To preload artifacts:

.. code-block:: bash

   mate download-mubin
   mate list-mubins

If only ``mate`` is installed, MATE downloads MUBIN artifacts on demand.

Delivered wrapper install
~~~~~~~~~~~~~~~~~~~~~~~~~

Supported delivered wrapper packages are ``flash_attn_3``, ``flash_mla``,
``flash_kda``, ``deep-gemm``, and ``sageattention``.

Before reinstalling a delivered package set, uninstall the packages you plan
to replace:

.. code-block:: bash

   python -m pip uninstall -y \
     mate flash_mla flash_kda deep-gemm sageattention flash_attn_3

Install the wrapper package that matches your framework surface. ``pip``
installs the matching ``mate`` dependency automatically.

.. code-block:: bash

   python -m pip install flash_attn_3 \
     --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

Replace ``flash_attn_3`` with ``flash_mla``, ``flash_kda``, ``deep-gemm``, or
``sageattention`` when those package surfaces match your framework.

Delivered MUSA wrapper versions use the PEP 440 local version suffix
``+musa``, for example ``0.2.4+musa``. Check the installed distribution to
distinguish the MATE-backed MUSA wrapper from the native implementation:

.. code-block:: bash

   python -m pip show flash_attn_3

The ``Version`` field should end in ``+musa``. Replace ``flash_attn_3`` with
the wrapper package you installed when checking another wrapper. The matching
``mate`` dependency keeps its normal version; ``+musa`` is a wrapper-only
identifier and does not encode the MUSA Toolkit version.

Direct MATE install
~~~~~~~~~~~~~~~~~~~

Install ``mate`` directly when no wrapper matches your workload or when you
want direct MATE Python APIs without a wrapper package.

.. code-block:: bash

   python -m pip install mate \
     --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple

Build from source
~~~~~~~~~~~~~~~~~

Get the source checkout
^^^^^^^^^^^^^^^^^^^^^^^

Clone the repository with submodules:

.. code-block:: bash

   git clone https://github.com/MooreThreads/mate.git --recursive
   cd mate

.. note::

   If the repository was cloned without ``--recursive``, run
   ``git submodule update --init --recursive`` in the repository root before
   building.

Install from the local checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For local builds, keep dependency resolution disabled so ``pip`` does not
replace the MUSA PyTorch stack with upstream PyPI packages.

- Use ``--no-build-isolation`` for source installs.
- Use ``--no-isolation`` for local wheel builds.
- Use ``--no-deps`` when installing local builds.

Choose one local install mode:

- Use an editable install when you are iterating on the local checkout.
- Build and install a local wheel when you need a built artifact.

Editable install
""""""""""""""""

Use an editable install when you are iterating on the local checkout:

.. code-block:: bash

   python -m pip install --no-build-isolation --no-deps -e . -v

Local wheel install
"""""""""""""""""""

Use a local wheel when you want a built artifact instead of an editable
install:

.. code-block:: bash

   python -m build --wheel --no-isolation
   python -m pip install --no-deps dist/mate-*.whl

Optional MUBIN wheel
""""""""""""""""""""

Build the optional package containing the complete pre-generated MUBIN payload
after installing the local MATE checkout:

.. code-block:: bash

   cd mate-mubin
   python -m build --no-isolation --wheel
   python -m pip install --no-deps dist/mate_mubin-*.whl

By default, the build backend downloads and verifies all artifacts pinned by
the current MATE source. To build from an existing complete cache, set
``MATE_MUBIN_SOURCE_DIR`` to its absolute root path before invoking the build.

MUBIN artifact availability
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MATE uses external MUBIN artifacts for selected ``gemm``,
``flash_attention``, ``flash_mla``, and ``sage_attention`` execution paths.
The public Python APIs remain in ``mate``, but those paths require one of the
following artifact sources at runtime:

- An installed ``mate-mubin`` package.
- A downloaded artifact cache, selected by ``MATE_MUBIN_DIR`` and defaulting
  to ``~/.cache/mate/mubin``.

An installed ``mate-mubin`` package takes precedence over the downloaded
cache and its contents are trusted at runtime. The main ``mate`` source build
and wheel do not themselves contain the external MUBIN payload.

Install the optional wheel that matches the running MATE version with:

.. code-block:: bash

   mate install-mubin-wheel

This command installs ``mate-mubin`` from the MUSA wheel source with
``--no-deps``. Use ``mate install-mubin-wheel --dry-run`` to inspect the exact
pip command, or pass ``--index-url`` when the optional wheel is hosted on a
different Python package index.

When ``mate-mubin`` is not installed, first use requires access to the
configured artifact repository. MATE first downloads ``kernel_map.json`` for
the selected module. It then downloads the required ``.o`` kernel object
lazily. All downloaded files are verified by default.

Use ``mate download-mubin`` when the complete payload must be available before
the first operator call, such as when preparing an offline host or container
image. See :doc:`Command Line Interface <mate_cli>` for command behavior and
:doc:`Environment Variables <environment_variables>` for cache and repository
controls.

Step 4. Validate Installation
-----------------------------

After installing MATE directly or through a wrapper package, validate the
MATE runtime first.

.. code-block:: bash

   python - <<'PY'
   import mate
   print("mate import ok")
   PY

.. code-block:: bash

   python -m mate --help
   mate check
   mate show-config
   mate env
   mate list-mubins

If the ``mate`` executable entrypoint is not available in your environment,
use ``python -m mate ...`` for supported subcommands.

If you installed a wrapper package in Step 3, follow that wrapper page for the
wrapper import path and package-specific validation snippet.

Prepare an offline environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the ``mate-mubin`` wheel is unavailable, populate a dedicated artifact
directory while repository access is available:

.. code-block:: bash

   export MATE_MUBIN_DIR="$HOME/mate-mubin-cache"
   mate download-mubin
   mate list-mubins

Confirm that every required module reports ``Downloaded``. Preserve that
directory in the offline environment, keep ``MATE_MUBIN_DIR`` set to the same
path, and then disable runtime retrieval:

.. code-block:: bash

   export MATE_MUBIN_DIR="$HOME/mate-mubin-cache"
   export MATE_MUBIN_NO_DOWNLOAD=1

A ``Metadata only`` status is insufficient for an offline workload that selects
a kernel which has not already been fetched. Do not disable kernel-map and
object hash verification as part of the offline workflow.

Step 5. Build a Wrapper from Source if Needed
---------------------------------------------

Skip this step if you installed a delivered wrapper package in Step 3.

Use this path only when you are developing a wrapper locally from the
repository checkout. Install local ``mate`` from the Step 3 source-build path
first. The commands below are for local editable installs or local wheel
installs, not the delivered wheel source.

- Install local ``mate`` first.
- Use ``--no-deps`` for local wrapper installs.

.. list-table::
   :header-rows: 1

   * - Wrapper directory
     - Package name
     - Import path
     - Typical use
   * - ``wrappers/flash-attention``
     - ``flash_attn_3``
     - ``flash_attn_interface``
     - FlashAttention-3 style integration
   * - ``wrappers/FlashMLA``
     - ``flash_mla``
     - ``flash_mla``
     - FlashMLA style integration
   * - ``wrappers/MSA``
     - ``fmha_sm100``
     - ``fmha_sm100``
     - MSA fmha_sm100 style integration
   * - ``wrappers/FlashKDA``
     - ``flash_kda``
     - ``flash_kda``
     - FlashKDA style integration
   * - ``wrappers/DeepGEMM``
     - ``deep-gemm``
     - ``deep_gemm``
     - DeepGEMM style integration
   * - ``wrappers/SageAttention``
     - ``sageattention``
     - ``sageattention``
     - SageAttention style integration

Editable install pattern:

.. code-block:: bash

   cd /path/to/mate
   python -m pip install --no-build-isolation --no-deps -e . -v
   cd wrappers/flash-attention
   python -m pip install --no-build-isolation --no-deps -e .

Wheel install pattern:

.. code-block:: bash

   cd /path/to/mate
   python -m pip install --no-build-isolation --no-deps -e . -v
   cd wrappers/flash-attention
   python -m build --wheel
   python -m pip install --no-deps dist/flash_attn_3-*.whl

Repeat the same workflow for ``wrappers/FlashMLA``, ``wrappers/MSA``,
``wrappers/FlashKDA``, ``wrappers/DeepGEMM``, and ``wrappers/SageAttention``
when those package surfaces match your framework.

Optional Step 6. Pre-Build AOT Kernels
--------------------------------------

If you want to pre-build AOT kernels before producing a wheel, run:

.. code-block:: bash

   MATE_MUSA_ARCH_LIST=3.1 python -m mate.aot
   python -m build --wheel --no-isolation

Customize AOT coverage when needed:

.. code-block:: bash

   python -m mate.aot --attention-aot-level 0 --add-gemm true --add-moe false

The AOT build command does not prefetch external MUBIN artifacts. Run
``mate download-mubin`` separately when an offline deployment also uses
MUBIN-backed operator paths.

Next Steps
----------

- Continue with :doc:`wrapper_tutorials` for wrapper-specific quickstarts.
- Continue with :doc:`diagnostics` if validation or runtime behavior fails.
- Continue with :doc:`api_reference` when no wrapper matches your workload.
