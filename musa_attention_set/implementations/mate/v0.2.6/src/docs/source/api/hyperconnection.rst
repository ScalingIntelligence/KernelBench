.. _apihyperconnection:

HyperConnection
===============

Use these MATE HyperConnection APIs when wrapper coverage is not enough.
For DeepGEMM-style prenorm GEMM flows, prefer the ``deep-gemm`` wrapper when it
matches your framework surface.

.. currentmodule:: mate.hyperconnection

MHC Pre
-------

.. autofunction:: mhc_pre
.. autofunction:: mhc_prenorm_gemm_sqrsum
.. autofunction:: mhc_pre_big_fuse
