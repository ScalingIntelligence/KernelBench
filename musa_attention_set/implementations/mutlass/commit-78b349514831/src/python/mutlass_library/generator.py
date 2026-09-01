#################################################################################################
#
# Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
# Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Utilities for enumerating MUTLASS library kernels
"""

import argparse
import enum
from itertools import product
import logging
import os.path
import shutil

import sys


# Certain usecases of mutlass_library nearly always prefer to run as scripts with
# relative imports, rather than via an installed Python package. An example of this
# is using MUTLASS's CMake system to generate a library of kernels to be profiled.
# To make it easy to use these use cases when an existing installation of mutlass_library
# exists, this global flag can be set to true (via command-line arguments) to ensure
# that package-based installations are not used.

# Create a temporary argument parser to check only for the availability of the
# --disable-mutlass-package-imports argument, which controls whether package-based
# imports are disabled.
def _add_package_disablement_flag(argparser):
  argparser.add_argument("--disable-mutlass-package-imports", action='store_true', required=False,
                     help="Disable use of mutlass_library from Python package")

_parser = argparse.ArgumentParser()
_add_package_disablement_flag(_parser)
_args, _ = _parser.parse_known_args()

# Add `MUTLASS_IGNORE_PACKAGE` to `builtins` so that it is visible for gating future
# imports without requiring importing another module. Ideally, we would just place this
# as a global variable in a module to that could be imported and checked (e.g.,
# utils.MUTLASS_IGNORE_PACKAGE). However, this raises the issue of determining
# where this module should be sourced (from the mutlass_library package or from
# a relative import), which is the problem this variable is being used to solve in the
# first place.
import builtins
builtins.MUTLASS_IGNORE_PACKAGE = _args.disable_mutlass_package_imports

try:
  if MUTLASS_IGNORE_PACKAGE:
    raise ImportError("Disabling attempt to import mutlass_library")
  from mutlass_library.library import *
  from mutlass_library.manifest import *
except ImportError:
  from library import *
  from manifest import *
###################################################################################################
#
def EpilogueAlignment(max_alignment, tile, epilogue_steps = 8):
  ''' Helper to compute the maximum alignment of the epilogue '''

  def product(X, identity = 1):
    result = identity
    for item in X:
      result *= item
    return result

  elements_per_thread = product(tile.threadblock_shape[:-1]) // product(tile.warp_count) // 32 // epilogue_steps
  return min(max_alignment, elements_per_thread)

def DefaultSwizzlingFunctor():
    return SwizzlingFunctor.Identity8
    # To use StreamK decomposition for basic GEMMs, set `swizzling_functor = SwizzlingFunctor.StreamK`

# Generates 3.0 API based GemmUniversal API kernels. Alignment constraints are folded in with layouts
def CreateGemmUniversal3xOperator(
    manifest, layouts, tile_descriptions, data_types,
    schedules = [[KernelScheduleType.ScheduleAuto, EpilogueScheduleType.ScheduleAuto]],
    epilogue_functor=EpilogueFunctor.LinearCombination,
    swizzling_functor=SwizzlingFunctor.Identity1,
    tile_schedulers=[TileSchedulerType.Default]):


  for s in schedules:
    assert(len(s) == 2)


  operations = []

  # by default, only generate the largest tile and largest alignment
  # if manifest.kernel_filter == '':
  #   tile_descriptions = [tile_descriptions[0]]

  combinations = product(layouts, tile_descriptions, data_types, schedules, tile_schedulers)
  for layout, tile_description, data_type, schedules, tile_scheduler in combinations:
    kernel_schedule, epilogue_schedule = schedules
    A = TensorDescription(
        data_type["a_type"], layout[0][0], layout[0][1])
    B = TensorDescription(
        data_type["b_type"], layout[1][0], layout[1][1])

    C = TensorDescription(data_type["c_type"], layout[2][0], layout[2][1])
    D = TensorDescription(data_type["d_type"], layout[2][0], layout[2][1])

    extra_args = {}
    gemm_kind = GemmKind.Universal3x
    element_compute = data_type.get("epi_type", data_type["acc_type"])

    operation = GemmOperation(
        gemm_kind, tile_description.minimum_compute_capability,
        tile_description, A, B, C, element_compute, epilogue_functor, swizzling_functor, D,
        kernel_schedule, epilogue_schedule, tile_scheduler, extra_args)

    manifest.append(operation)
    operations.append(operation)

  return operations

def GenerateMP22_Simt_gemm_f32(manifest, musa_version):
  math_inst = MathInstruction(
                [1, 1, 1],
                DataType.f32, DataType.f32, DataType.f32,
                OpcodeClass.Simt)

  min_cc = 22
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128,  32, 4], 2, math_inst, min_cc, max_cc),
    TileDescription([128,  64, 4], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 4], 2, math_inst, min_cc, max_cc),

    TileDescription([128,  64, 8], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 8], 2, math_inst, min_cc, max_cc),
  ]

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }
  ]

  schedules_default = [
    [KernelScheduleType.Multistage, EpilogueScheduleType.ScheduleAuto],
  ]

  for tile_description in tile_descriptions:
    tile_m, tile_n, tile_k = tile_description.tile_shape

    thread_num_ = max(tile_m * tile_n // 64, 128)
    thread_num = thread_num_ * max((tile_m * tile_n * tile_k) // thread_num_ // 256, 1)
    aligns = [
      [
        min(tile_m * tile_k // thread_num, 4),
        min(tile_n * tile_k // thread_num, 4),
        min(tile_m * tile_n // thread_num, 4)
      ],
      [1, 1, 1]
    ]
    for align_a, align_b, align_c in aligns:

      layouts = [
        [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
        [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      ]

      CreateGemmUniversal3xOperator(manifest, layouts, [tile_description], data_types, schedules_default)

def GenerateMP22_TensorOp_gemm_tf32(manifest, musa_version):
  math_inst = MathInstruction(
                [32, 32, 8],
                DataType.tf32, DataType.tf32, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 22
  max_cc = 22

  tile_descriptions = [
    TileDescription([128, 32, 16], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 64, 16], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 16], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 16], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 16], 2, math_inst, min_cc, max_cc),
  ]

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }
  ]

  schedules_default = [
    [KernelScheduleType.Multistage, EpilogueScheduleType.ScheduleAuto],
  ]

  aligns = [
    [4, 4, 4],
    [1, 1, 1],
  ]

  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types, schedules_default)

def GenerateMP22_TensorOp_gemm_f16(manifest, musa_version):
  math_inst = MathInstruction(
                [32, 32, 16],
                DataType.f16, DataType.f16, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 22
  max_cc = 22

  tile_descriptions = [
    TileDescription([128, 32, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 64, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 32], 2, math_inst, min_cc, max_cc),
  ]

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }
  ]

  schedules_default = [
    [KernelScheduleType.Multistage, EpilogueScheduleType.ScheduleAuto],
  ]

  aligns = [
    [8, 8, 8],
    [1, 1, 1],

  ]

  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types, schedules_default)

def GenerateMP22_TensorOp_gemm_bf16(manifest, musa_version):
  math_inst = MathInstruction(
                [32, 32, 16],
                DataType.bf16, DataType.bf16, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 22
  max_cc = 22

  tile_descriptions = [
    TileDescription([128, 32, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 64, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 32], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 32], 2, math_inst, min_cc, max_cc),
  ]

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }
  ]

  schedules_default = [
    [KernelScheduleType.Multistage, EpilogueScheduleType.ScheduleAuto],
  ]

  aligns = [
    [8, 8, 8],
    [1, 1, 1],
  ]

  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types, schedules_default)

def GenerateMP22_TensorOp_gemm_s8(manifest, musa_version):
  math_inst = MathInstruction(
      [32, 32, 32],
      DataType.s8, DataType.s8, DataType.s32,
      OpcodeClass.TensorOp)

  min_cc = 22
  max_cc = 22

  tile_descriptions = [
    TileDescription([128, 32, 64], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 64, 64], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 64], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 64], 2, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 64], 2, math_inst, min_cc, max_cc),
  ]

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }
  ]

  schedules_default = [
    [KernelScheduleType.Multistage, EpilogueScheduleType.ScheduleAuto],
  ]

  aligns = [
    [16, 16, 16],
    [1,  1,  1],

  ]

  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types, schedules_default)

def GenerateMP31_Simt_gemm_f32(manifest, musa_version):
  math_inst = MathInstruction(
                [1, 1, 1],
                DataType.f32, DataType.f32, DataType.f32,
                OpcodeClass.Simt)

  min_cc = 31
  max_cc = 1024

  tile_descriptions = [
    TileDescription([128,  32, 8], 2, math_inst, min_cc, max_cc),
    TileDescription([128,  64, 8], 2, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 8], 2, math_inst, min_cc, max_cc),
  ]

  data_types = [
    {
      "a_type"   : math_inst.element_a,
      "b_type"   : math_inst.element_b,
      "c_type"   : math_inst.element_accumulator,
      "d_type"   : math_inst.element_accumulator,
      "acc_type" : math_inst.element_accumulator,
      "epi_type" : math_inst.element_accumulator
    }
  ]

  schedules_default = [
    [KernelScheduleType.Multistage, EpilogueScheduleType.ScheduleAuto],
  ]

  for tile_description in tile_descriptions:
    tile_m, tile_n, tile_k = tile_description.tile_shape
    thread_num_ = max(tile_m * tile_n // 64, 128)
    thread_num = thread_num_ * max((tile_m * tile_n * tile_k) // thread_num_ // 256, 1)
    aligns = [
      [
        min(tile_m * tile_k // thread_num, 4),
        min(tile_n * tile_k // thread_num, 4),
        min(tile_m * tile_n // thread_num, 4)
      ],
      [1, 1, 1]
    ]
    for align_a, align_b, align_c in aligns:

      layouts = [
        [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
        [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
        [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
        [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      ]

      CreateGemmUniversal3xOperator(manifest, layouts, [tile_description], data_types, schedules_default)

def GenerateMP31_TensorOp_gemm_tf32(manifest, musa_version):
  math_inst = MathInstruction(
                [0, 0, 0],
                DataType.tf32, DataType.tf32, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 31
  max_cc = 31

  both_k_major_tile_descriptions = [
    TileDescription([128, 64,  32], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 5, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 16], 4, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 16], 5, math_inst, min_cc, max_cc),
    TileDescription([384, 256, 16], 4, math_inst, min_cc, max_cc),
  ]
  not_all_k_major_tile_descriptions = [
    TileDescription([128, 64,  32], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 32], 5, math_inst, min_cc, max_cc),
    TileDescription([192, 128, 16], 4, math_inst, min_cc, max_cc),
  ]

  cd_types = [
    (math_inst.element_accumulator, math_inst.element_accumulator),
    (DataType.void, math_inst.element_accumulator),
  ]
  data_type = {
    "a_type"   : math_inst.element_a,
    "b_type"   : math_inst.element_b,
    "c_type"   : math_inst.element_accumulator,
    "d_type"   : math_inst.element_accumulator,
    "acc_type" : math_inst.element_accumulator,
    "epi_type" : math_inst.element_accumulator
  }

  schedules_default = [
    [KernelScheduleType.Tme, EpilogueScheduleType.WithTme],
    [KernelScheduleType.TmeWarpSpecialized, EpilogueScheduleType.NoSmem],
  ]

  aligns = [
    [1, 1, 1],
  ]
  data_types = []
  for cd_type in cd_types:
    data_type_ = data_type.copy()
    data_type_["c_type"] = cd_type[0]
    data_type_["d_type"] = cd_type[1]
    data_types.append(data_type_)
  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    for layout in layouts:
      if layout[0][0] == LayoutType.RowMajor and layout[1][0] == LayoutType.ColumnMajor:
        for tile_description in both_k_major_tile_descriptions:
          tile_description.math_instruction.instruction_shape = [
            min(128, tile_description.threadblock_shape[0]),
            min(128, tile_description.threadblock_shape[1]),
            min(32, tile_description.threadblock_shape[2]),
          ]
          for schedule in schedules_default:
            if schedule[0] == KernelScheduleType.TmeWarpSpecialized and tile_description.threadblock_shape[0] > 256:
              continue
            CreateGemmUniversal3xOperator(manifest, [layout], [tile_description], data_types, [schedule])
      else:
        for tile_description in not_all_k_major_tile_descriptions:
          tile_description.math_instruction.instruction_shape = [
            min(64, tile_description.threadblock_shape[0]),
            min(64, tile_description.threadblock_shape[1]),
            min(32, tile_description.threadblock_shape[2]),
          ]
        CreateGemmUniversal3xOperator(manifest, [layout], [tile_description], data_types, schedules_default)

def GenerateMP31_TensorOp_gemm_f16(manifest, musa_version):
  math_inst = MathInstruction(
                [0, 0, 0],
                DataType.f16, DataType.f16, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 31
  max_cc = 31

  tile_descriptions = [
    TileDescription([128, 64,  64], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 64], 5, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 32], 4, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 32], 5, math_inst, min_cc, max_cc),
    TileDescription([384, 256, 32], 4, math_inst, min_cc, max_cc),
  ]

  for tile_description in tile_descriptions:
    tile_description.math_instruction.instruction_shape = [
      min(128, tile_description.threadblock_shape[0]),
      min(128, tile_description.threadblock_shape[1]),
      min(64, tile_description.threadblock_shape[2]),
    ]
  cd_types = [
    (math_inst.element_accumulator, math_inst.element_accumulator),
    (math_inst.element_a, math_inst.element_a),
    (DataType.void, math_inst.element_accumulator),
    (DataType.void, math_inst.element_a),
  ]
  data_type = {
    "a_type"   : math_inst.element_a,
    "b_type"   : math_inst.element_b,
    "c_type"   : math_inst.element_accumulator,
    "d_type"   : math_inst.element_accumulator,
    "acc_type" : math_inst.element_accumulator,
    "epi_type" : math_inst.element_accumulator
  }


  schedules_default = [
    [KernelScheduleType.Tme, EpilogueScheduleType.WithTme],
    [KernelScheduleType.TmeWarpSpecialized, EpilogueScheduleType.NoSmem],
  ]

  aligns = [
    [2, 2, 2],
  ]
  data_types = []
  for cd_type in cd_types:
    data_type_ = data_type.copy()
    data_type_["c_type"] = cd_type[0]
    data_type_["d_type"] = cd_type[1]
    data_types.append(data_type_)
  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]
    
    for schedule in schedules_default:
        for tile_description in tile_descriptions:
          if schedule[0] == KernelScheduleType.TmeWarpSpecialized and tile_description.threadblock_shape[0] > 256:
            continue
          CreateGemmUniversal3xOperator(manifest, layouts, [tile_description], data_types, [schedule])

def GenerateMP31_TensorOp_gemm_bf16(manifest, musa_version):
  math_inst = MathInstruction(
                [0, 0, 0],
                DataType.bf16, DataType.bf16, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 31
  max_cc = 31

  tile_descriptions = [
    TileDescription([128, 64,  64], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 64], 5, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 32], 4, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 32], 5, math_inst, min_cc, max_cc),
    TileDescription([384, 256, 32], 4, math_inst, min_cc, max_cc),
  ]

  for tile_description in tile_descriptions:
    tile_description.math_instruction.instruction_shape = [
      min(128, tile_description.threadblock_shape[0]),
      min(128, tile_description.threadblock_shape[1]),
      min(64, tile_description.threadblock_shape[2]),
    ]

  cd_types = [
    (math_inst.element_accumulator, math_inst.element_accumulator),
    (math_inst.element_a, math_inst.element_a),
    (DataType.void, math_inst.element_accumulator),
    (DataType.void, math_inst.element_a),
  ]
  data_type = {
    "a_type"   : math_inst.element_a,
    "b_type"   : math_inst.element_b,
    "c_type"   : math_inst.element_accumulator,
    "d_type"   : math_inst.element_accumulator,
    "acc_type" : math_inst.element_accumulator,
    "epi_type" : math_inst.element_accumulator
  }

  schedules_default = [
    [KernelScheduleType.Tme, EpilogueScheduleType.WithTme],
    [KernelScheduleType.TmeWarpSpecialized, EpilogueScheduleType.NoSmem],
  ]

  aligns = [
    [2, 2, 2],
  ]

  data_types = []
  for cd_type in cd_types:
    data_type_ = data_type.copy()
    data_type_["c_type"] = cd_type[0]
    data_type_["d_type"] = cd_type[1]
    data_types.append(data_type_)
  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    for schedule in schedules_default:
        for tile_description in tile_descriptions:
          if schedule[0] == KernelScheduleType.TmeWarpSpecialized and tile_description.threadblock_shape[0] > 256:
            continue
          CreateGemmUniversal3xOperator(manifest, layouts, [tile_description], data_types, [schedule])


def GenerateMP31_TensorOp_gemm_e4m3(manifest, musa_version):
  math_inst = MathInstruction(
                [0, 0, 0],
                DataType.e4m3, DataType.e4m3, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 31
  max_cc = 31

  tile_descriptions = [
    TileDescription([256, 128, 64], 4, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 64], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 256, 128], 4, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 64], 4, math_inst, min_cc, max_cc),
    TileDescription([64, 128, 64], 4, math_inst, min_cc, max_cc),
  ]

  for tile_description in tile_descriptions:
    tile_description.math_instruction.instruction_shape = [
      min(128, tile_description.threadblock_shape[0]),
      min(128, tile_description.threadblock_shape[1]),
      min(64, tile_description.threadblock_shape[2]),
    ]

  cd_types = [
    (math_inst.element_accumulator, math_inst.element_accumulator),
    (DataType.void, math_inst.element_accumulator),
  ]
  data_type = {
    "a_type"   : math_inst.element_a,
    "b_type"   : math_inst.element_b,
    "c_type"   : math_inst.element_accumulator,
    "d_type"   : math_inst.element_accumulator,
    "acc_type" : math_inst.element_accumulator,
    "epi_type" : math_inst.element_accumulator
  }

  schedules_default = [
    [KernelScheduleType.Tme, EpilogueScheduleType.WithTme],
    [KernelScheduleType.TmeWarpSpecialized, EpilogueScheduleType.NoSmem],
  ]

  aligns = [
    [4, 4, 4],
  ]

  data_types = []
  for cd_type in cd_types:
    data_type_ = data_type.copy()
    data_type_["c_type"] = cd_type[0]
    data_type_["d_type"] = cd_type[1]
    data_types.append(data_type_)
  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types, schedules_default)
  
def GenerateMP31_TensorOp_gemm_e5m2(manifest, musa_version):
  math_inst = MathInstruction(
                [0, 0, 0],
                DataType.e5m2, DataType.e5m2, DataType.f32,
                OpcodeClass.TensorOp)

  min_cc = 31
  max_cc = 31

  tile_descriptions = [
    TileDescription([256, 128, 64], 4, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 64], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 256, 128], 4, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 64], 4, math_inst, min_cc, max_cc),
    TileDescription([128, 64, 64], 4, math_inst, min_cc, max_cc),
  ]

  for tile_description in tile_descriptions:
    tile_description.math_instruction.instruction_shape = [
      min(128, tile_description.threadblock_shape[0]),
      min(128, tile_description.threadblock_shape[1]),
      min(64, tile_description.threadblock_shape[2]),
    ]

  cd_types = [
    (math_inst.element_accumulator, math_inst.element_accumulator),
    (DataType.void, math_inst.element_accumulator),
  ]
  data_type = {
    "a_type"   : math_inst.element_a,
    "b_type"   : math_inst.element_b,
    "c_type"   : math_inst.element_accumulator,
    "d_type"   : math_inst.element_accumulator,
    "acc_type" : math_inst.element_accumulator,
    "epi_type" : math_inst.element_accumulator
  }

  schedules_default = [
    [KernelScheduleType.Tme, EpilogueScheduleType.WithTme],
    [KernelScheduleType.TmeWarpSpecialized, EpilogueScheduleType.NoSmem],
  ]

  aligns = [
    [4, 4, 4],
  ]

  data_types = []
  for cd_type in cd_types:
    data_type_ = data_type.copy()
    data_type_["c_type"] = cd_type[0]
    data_type_["d_type"] = cd_type[1]
    data_types.append(data_type_)
  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    CreateGemmUniversal3xOperator(manifest, layouts, tile_descriptions, data_types, schedules_default)

def GenerateMP31_TensorOp_gemm_s8(manifest, musa_version):
  math_inst = MathInstruction(
                [0, 0 , 0],
                DataType.s8, DataType.s8, DataType.s32,
                OpcodeClass.TensorOp)

  min_cc = 31
  max_cc = 31

  tile_descriptions = [
    TileDescription([128, 64,  128], 3, math_inst, min_cc, max_cc),
    TileDescription([128, 128, 128], 5, math_inst, min_cc, max_cc),
    TileDescription([256, 128, 64], 4, math_inst, min_cc, max_cc),
    TileDescription([256, 256, 64], 5, math_inst, min_cc, max_cc),
    TileDescription([384, 256, 64], 4, math_inst, min_cc, max_cc),
  ]

  for tile_description in tile_descriptions:
    tile_description.math_instruction.instruction_shape = [
      min(128, tile_description.threadblock_shape[0]),
      min(128, tile_description.threadblock_shape[1]),
      min(128, tile_description.threadblock_shape[2]),
    ]

  cd_types = [
    (math_inst.element_accumulator, math_inst.element_accumulator),
    (DataType.void, math_inst.element_accumulator),
  ]
  data_type = {
    "a_type"   : math_inst.element_a,
    "b_type"   : math_inst.element_b,
    "c_type"   : math_inst.element_accumulator,
    "d_type"   : math_inst.element_accumulator,
    "acc_type" : math_inst.element_accumulator,
    "epi_type" : math_inst.element_accumulator
  }

  schedules_default = [
    [KernelScheduleType.Tme, EpilogueScheduleType.WithTme],
    [KernelScheduleType.TmeWarpSpecialized, EpilogueScheduleType.NoSmem],
  ]

  aligns = [
    [8, 8, 8],
  ]

  data_types = []
  for cd_type in cd_types:
    data_type_ = data_type.copy()
    data_type_["c_type"] = cd_type[0]
    data_type_["d_type"] = cd_type[1]
    data_types.append(data_type_)

  for align_a, align_b, align_c in aligns:
    layouts = [
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.ColumnMajor, align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.RowMajor,    align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.ColumnMajor, align_b], [LayoutType.RowMajor,    align_c]],
      [[LayoutType.ColumnMajor, align_a], [LayoutType.RowMajor,    align_b], [LayoutType.RowMajor,    align_c]],
    ]

    for schedule in schedules_default:
        for tile_description in tile_descriptions:
          if schedule[0] == KernelScheduleType.TmeWarpSpecialized and tile_description.threadblock_shape[0] > 256:
            continue
          CreateGemmUniversal3xOperator(manifest, layouts, [tile_description], data_types, [schedule])



#
def GenerateMP22(manifest, musa_version):
  GenerateMP22_Simt_gemm_f32(manifest, musa_version)
  GenerateMP22_TensorOp_gemm_tf32(manifest, musa_version)
  GenerateMP22_TensorOp_gemm_f16(manifest, musa_version)
  GenerateMP22_TensorOp_gemm_bf16(manifest, musa_version)
  GenerateMP22_TensorOp_gemm_s8(manifest, musa_version)

def GenerateMP31(manifest, musa_version):
  # GenerateMP31_Simt_gemm_f32(manifest, musa_version)
  GenerateMP31_TensorOp_gemm_tf32(manifest, musa_version)
  GenerateMP31_TensorOp_gemm_f16(manifest, musa_version)
  GenerateMP31_TensorOp_gemm_bf16(manifest, musa_version)
  GenerateMP31_TensorOp_gemm_e4m3(manifest, musa_version)
  GenerateMP31_TensorOp_gemm_e5m2(manifest, musa_version)
  GenerateMP31_TensorOp_gemm_s8(manifest, musa_version)

###################################################################################################

def numeric_log_level(log_level: str) -> int:
  """
  Converts the string identifier of the log level into the numeric identifier used
  in setting the log level

  :param x: string representation of log level (e.g., 'INFO', 'DEBUG')
  :type x: str

  :return: numeric representation of log level
  :rtype: int
  """
  numeric_level = getattr(logging, log_level.upper(), None)
  if not isinstance(numeric_level, int):
    raise ValueError(f'Invalid log level: {log_level}')
  return numeric_level

# This function for defining the ArgumentParser is used to make it easy for the MUTLASS Python interface
# to leverage the functionality in this file without running this script via a shell prompt.
def define_parser():
  parser = argparse.ArgumentParser(description="Generates device kernel registration code for MUTLASS Kernels")
  parser.add_argument("--operations", default="gemm", help="Specifies the operation to generate (gemm, all)")
  parser.add_argument("--build-dir", default="../build", required=False, help="MUTLASS top-level build directory")
  parser.add_argument("--curr-build-dir", default="..//build/tools/library", help="MUTLASS current build directory. cmake files will be emitted in this directory")
  parser.add_argument("--generator-target", default='library', help="Target of MUTLASS Library Generator.")
  parser.add_argument("--architectures", default='22', help="Target compute architectures")
  parser.add_argument("--kernels", default='', help='Comma delimited list to filter kernels by name.')
  parser.add_argument("--ignore-kernels", default='', help='Comma delimited list of kernels to exclude from build.')
  parser.add_argument("--filter-by-cc", default='True', type=str, help='If enabled, kernels whose compute capability range is not satisfied by the build target are excluded.')
  parser.add_argument("--musa-version", default="3.0.0", help="Semantic version string of MUSA Toolkit")
  parser.add_argument('--kernel-filter-file',   type=str, default=None, required=False, help='Full path of filter file')
  parser.add_argument('--selected-kernel-list',   type=str, default=None, required=False,
                        help='Specify the output log file containing all enabled kernels in this build')
  parser.add_argument("--disable-full-archs-compilation", action="store_true", required=False, help="Disable compilation for every archs in --architectures")
  parser.add_argument("--log-level", default='info', type=numeric_log_level, required=False,
                      help='Logging level to be used by the generator script')
  _add_package_disablement_flag(parser)
  return parser


if __name__ == "__main__":
  parser = define_parser()
  args = parser.parse_args()
  # Set the logging level based on the user-provided `--log-level` command-line option
  logging.basicConfig(level=args.log_level)

  manifest = Manifest(args)
  GenerateMP22(manifest, args.musa_version)
  GenerateMP31(manifest, args.musa_version)
  if 'library' in args.generator_target.split(','):
    manifest.emit(GeneratorTarget.Library)

  if args.selected_kernel_list is not None:
    if len(manifest.selected_kernels) > 0:
      with open(args.selected_kernel_list, 'w') as file_writer:
        for line in manifest.selected_kernels:
          file_writer.write("%s\n" % line)

###################################################################################################
