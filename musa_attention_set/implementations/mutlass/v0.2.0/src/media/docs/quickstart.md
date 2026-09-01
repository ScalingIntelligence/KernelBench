[README](../../README.md#documentation) > **Quick Start**

# Quickstart

## Prerequisites

MUTLASS requires:

- MooreThreads MUSA Toolkit (4.0.0 or later required)

- MCC 4.0.0 or later

- CMake 3.8+

- Python 3.5+


## Initial build steps

Construct a build directory and run CMake.

```bash
$ mkdir build && cd build
$ cmake .. -DMUTLASS_MCC_ARCHS=22             # compiles for MooreThreads Quyuan GPU architecture
```

If your goal is strictly to build only the MUTLASS Profiler and to minimize compilation time, we suggest executing the following CMake command in an empty `build/` directory.

```bash
$ cmake .. -DMUTLASS_MCC_ARCHS=22 -DMUTLASS_ENABLE_TESTS=OFF -DMUTLASS_ENABLE_EXAMPLES=OFF
```

This reduces overall compilation time by excluding unit tests and examples.

MUTLASS Profiler automatically generates a wide range of kernels, but you can control which kernels are generated using the `MUTLASS_LIBRARY_KERNELS` option. Additionally, the `MUTLASS_LIBRARY_IGNORE_KERNELS` option allows you to filter out unwanted kernels from the generated set. For example, if you only wish to generate the tensorop kernels and do not need the kernels with a tile shape of 256x128, you can execute the following command:

```bash
$ cmake .. -DMUTLASS_LIBRARY_KERNELS="tensorop" -DMUTLASS_LIBRARY_IGNORE_KERNELS="256x128"
```

## Build and run the MUTLASS Profiler

From the `build/` directory created above, compile the MUTLASS Profiler.

```bash
$ make mutlass_profiler -j12
```

Then execute the MUTLASS Profiler computing GEMM, execute the following command.

```bash
$ ./tools/profiler/mutlass_profiler --operation=Gemm --op_class=simt --m=4096 --n=4096 --k=1024 --cta_m=128 --cta_n=128

=============================
  Problem ID: 1

        Provider: MUTLASS
   OperationKind: gemm
       Operation: mutlass_mp22_simt_sgemm_f32_f32_f32_f32_f32_128x128x4_tnn_align2

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          muBLAS: Not run
           muDNN: Not run

       Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=1024 --A=f32:row --B=f32:column --C=f32:column --D=f32:column  \
                  --alpha=1 --beta=0 --batch_count=1 --op_class=simt --accum=f32 --cta_m=128 --cta_n=128 --cta_k=4 --cluster_m=1  \
                  --cluster_n=1 --cluster_k=1 --stages=2 --inst_m=1 --inst_n=1 --inst_k=1 --min_cc=22 --max_cc=1024

           Bytes: 100663296  bytes
           FLOPs: 34393292800  flops
           FLOPs/Byte: 341

         Runtime: 2.92589  ms
          Memory: 32.0416 GiB/s

            Math: 11754.8 GFLOP/s
```

You can use the following two commands to get more information:

```bash
$ ./tools/profiler/mutlass_profiler --help
$ ./tools/profiler/mutlass_profiler --operation=Gemm --help
```

## Build and run MUTLASS Unit Tests

From the `build/` directory created above, simply build the target `test_unit` to compile and run all unit tests.

```bash
$ make test_unit -j
...
...
...
[----------] Global test environment tear-down
[==========] 24 tests from 12 test suites ran. (11180 ms total)
[  PASSED  ] 24 tests.
```

The exact number of tests run is subject to change as we add more functionality.

No tests should fail. Unit tests automatically construct the appropriate runtime filters to avoid executing on architectures that do not support all features under test.

The unit tests are arranged hierarchically mirroring the MUTLASS Template Library. This enables parallelism in building and running tests as well as reducing compilation times when a specific set of tests are desired.

For example, the following executes strictly the MuTe.Core tests.

```bash
$ make test_unit_mute_core -j
...
...
[       OK ] MuTe_core.Tuple (0 ms)
[ RUN      ] MuTe_core.WeaklyCongruent
[       OK ] MuTe_core.WeaklyCongruent (0 ms)
[ RUN      ] MuTe_core.WeaklyCompatible
[       OK ] MuTe_core.WeaklyCompatible (0 ms)
[----------] 33 tests from MuTe_core (7 ms total)

[----------] Global test environment tear-down
[==========] 33 tests from 1 test suite ran. (7 ms total)
[  PASSED  ] 33 tests.
[100%] Built target test_unit_mute_core
```

## Building for Multiple Architectures

To minimize compilation time, specific GPU architectures can be enabled via the CMake command when we add support for more architectures in the future.

**MooreThreads Quyuan Architecture.**

```bash
cmake .. -DMUTLASS_MCC_ARCHS=22             # compiles for MooreThreads Quyuan GPU architecture
```

**MooreThreads MP31 Architecture.**

```bash
cmake .. -DMUTLASS_MCC_ARCHS=31             # compiles for MooreThreads MP31 GPU architecture
```

## Using MUTLASS within other applications

Applications should list [`/include`](/include) within their include paths. They must be compiled as C++17 or greater.

If the file extension is not `.mu`, you need to prefix the filename with `-x musa` to enable full capability. For example, `mcc -x musa test.cc -std=c++17`.

**Example:** print the contents of a variable storing half-precision data.

```c++
#include <iostream>
#include <mutlass/mutlass.h>
#include <mutlass/numeric_types.h>
#include <mutlass/core_io.h>
int main() {
  mutlass::half_t x = 2.25_hf;
  std::cout << x << std::endl;
  return 0;
}
```

## Launching a GEMM kernel in MUSA

Please refer to the examples for the usage of the low-level API and the collective builders: examples [00](/examples/00_basic_gemm) and [01](/examples/01_quyuan_gemm_with_collective_builder).

# MUTLASS Library

The [MUTLASS Library](/tools/library) defines an API for managing and executing collections of compiled kernel instances and launching them from host code without template instantiations in client code.

The host-side launch API is designed to be analogous to BLAS implementations for convenience, though its kernel selection procedure is intended only to be functionally sufficient. It may not launch the optimal tile size for a given problem. It chooses the first available kernel whose data types, layouts, and alignment constraints satisfy the given problem. Kernel instances and a data structure describing them are completely available to client applications which may choose to implement their own selection logic.

The MUTLASS Library is used by the MUTLASS Profiler to manage kernel instances.
