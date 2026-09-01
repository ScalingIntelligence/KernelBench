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
Data types and tags used for emitting MUTLASS C++ kernels
"""

import enum
import re
import copy

# The following block implements enum.auto() for Python 3.5 variants that don't include it such
# as the default 3.5.2 on Ubuntu 16.04.
#
# https://codereview.stackexchange.com/questions/177309/reimplementing-pythons-enum-auto-for-compatibility

try:
  from enum import auto as enum_auto
except ImportError:
  __mutlass_library_auto_enum = 0
  def enum_auto() -> int:
    global __mutlass_library_auto_enum
    i = __mutlass_library_auto_enum
    __mutlass_library_auto_enum += 1
    return i

###################################################################################################

#
class GeneratorTarget(enum.Enum):
  Library = enum_auto()
#
GeneratorTargetNames = {
  GeneratorTarget.Library: 'library'
}
#

###################################################################################################

#
class Underscore:
  def __str__(self):
    return "Underscore"


class DataType(enum.Enum):
  void = enum_auto()  # primarily used to disable C tensor for epilogues
  b1 = enum_auto()
  u2 = enum_auto()
  u4 = enum_auto()
  u8 = enum_auto()
  u16 = enum_auto()
  u32 = enum_auto()
  u64 = enum_auto()
  s2 = enum_auto()
  s4 = enum_auto()
  s8 = enum_auto()
  s16 = enum_auto()
  s32 = enum_auto()
  s64 = enum_auto()
  e4m3 = enum_auto()
  e5m2 = enum_auto()
  f8 = enum_auto()    
  f6 = enum_auto()    
  f4 = enum_auto()    
  e3m2 = enum_auto()     
  e2m3 = enum_auto()     
  e2m1 = enum_auto()     
  ue8m0 = enum_auto()    
  ue4m3 = enum_auto()    
  f16 = enum_auto()
  bf16 = enum_auto()
  f32 = enum_auto()
  tf32 = enum_auto()
  f64 = enum_auto()
  cf16 = enum_auto()
  cbf16 = enum_auto()
  cf32 = enum_auto()
  ctf32 = enum_auto()
  cf64 = enum_auto()
  cs2 = enum_auto()
  cs4 = enum_auto()
  cs8 = enum_auto()
  cs16 = enum_auto()
  cs32 = enum_auto()
  cs64 = enum_auto()
  cu2 = enum_auto()
  cu4 = enum_auto()
  cu8 = enum_auto()
  cu16 = enum_auto()
  cu32 = enum_auto()
  cu64 = enum_auto()
  invalid = enum_auto()

#
ShortDataTypeNames = {
  DataType.s32: 'i',
  DataType.e4m3: 'e4m3',
  DataType.e5m2: 'e5m2',
  DataType.f16: 'h',
  DataType.f32: 's',
  DataType.f64: 'd',
  DataType.cf32: 'c',
  DataType.cf64: 'z',
  DataType.f8: 'f8',      
  DataType.f6: 'f6',      
  DataType.f4: 'f4',
}

#
DataTypeNames = {
  DataType.void: "void",
  DataType.b1: "b1",
  DataType.u2: "u2",
  DataType.u4: "u4",
  DataType.u8: "u8",
  DataType.u16: "u16",
  DataType.u32: "u32",
  DataType.u64: "u64",
  DataType.s2: "s2",
  DataType.s4: "s4",
  DataType.s8: "s8",
  DataType.s16: "s16",
  DataType.s32: "s32",
  DataType.s64: "s64",
  DataType.e4m3: 'e4m3',
  DataType.e5m2: 'e5m2',
  DataType.f8: 'f8',     
  DataType.f6: 'f6',     
  DataType.f4: 'f4',     
  DataType.e2m3: 'e2m3',       
  DataType.e3m2: 'e3m2',       
  DataType.e2m1: 'e2m1',       
  DataType.ue8m0: 'ue8m0',     
  DataType.ue4m3: 'ue4m3',     
  DataType.f16: "f16",
  DataType.bf16: "bf16",
  DataType.f32: "f32",
  DataType.tf32: "tf32",
  DataType.f64: "f64",
  DataType.cf16: "cf16",
  DataType.cbf16: "cbf16",
  DataType.cf32: "cf32",
  DataType.ctf32: "ctf32",
  DataType.cf64: "cf64",
  DataType.cu2: "cu2",
  DataType.cu4: "cu4",
  DataType.cu8: "cu8",
  DataType.cu16: "cu16",
  DataType.cu32: "cu32",
  DataType.cu64: "cu64",
  DataType.cs2: "cs2",
  DataType.cs4: "cs4",
  DataType.cs8: "cs8",
  DataType.cs16: "cs16",
  DataType.cs32: "cs32",
  DataType.cs64: "cs64",
}

DataTypeTag = {
  DataType.void: "void",
  DataType.b1: "mutlass::uint1b_t",
  DataType.u2: "mutlass::uint2b_t",
  DataType.u4: "mutlass::uint4b_t",
  DataType.u8: "uint8_t",
  DataType.u16: "uint16_t",
  DataType.u32: "uint32_t",
  DataType.u64: "uint64_t",
  DataType.s2: "mutlass::int2b_t",
  DataType.s4: "mutlass::int4b_t",
  DataType.s8: "int8_t",
  DataType.s16: "int16_t",
  DataType.s32: "int32_t",
  DataType.s64: "int64_t",
  DataType.e4m3: 'mutlass::float_e4m3_t',
  DataType.e5m2: 'mutlass::float_e5m2_t',
  DataType.f8: 'mutlass::type_erased_dynamic_float8_t',      
  DataType.f6: 'mutlass::type_erased_dynamic_float6_t',      
  DataType.f4: 'mutlass::type_erased_dynamic_float4_t',      
  DataType.e2m3: 'mutlass::float_e2m3_t',                       
  DataType.e3m2: 'mutlass::float_e3m2_t',                       
  DataType.e2m1: 'mutlass::float_e2m1_t',                       
  DataType.ue8m0: 'mutlass::float_ue8m0_t',                     
  DataType.ue4m3: 'mutlass::float_ue4m3_t',                     
  DataType.f16: "mutlass::half_t",
  DataType.bf16: "mutlass::bfloat16_t",
  DataType.f32: "float",
  DataType.tf32: "mutlass::tfloat32_t",
  DataType.f64: "double",
  DataType.cf16: "mutlass::complex<mutlass::half_t>",
  DataType.cbf16: "mutlass::complex<mutlass::bfloat16_t>",
  DataType.cf32: "mutlass::complex<float>",
  DataType.ctf32: "mutlass::complex<mutlass::tfloat32_t>",
  DataType.cf64: "mutlass::complex<double>",
  DataType.cu2: "mutlass::complex<mutlass::uint2b_t>",
  DataType.cu4: "mutlass::complex<mutlass::uint4b_t>",
  DataType.cu8: "mutlass::complex<mutlass::uint8_t>",
  DataType.cu16: "mutlass::complex<mutlass::uint16_t>",
  DataType.cu32: "mutlass::complex<mutlass::uint32_t>",
  DataType.cu64: "mutlass::complex<mutlass::uint64_t>",
  DataType.cs2: "mutlass::complex<mutlass::int2b_t>",
  DataType.cs4: "mutlass::complex<mutlass::int4b_t>",
  DataType.cs8: "mutlass::complex<mutlass::int8_t>",
  DataType.cs16: "mutlass::complex<mutlass::int16_t>",
  DataType.cs32: "mutlass::complex<mutlass::int32_t>",
  DataType.cs64: "mutlass::complex<mutlass::int64_t>",
}

DataTypeSize = {
  DataType.void: 0,
  DataType.b1: 1,
  DataType.u2: 2,
  DataType.u4: 4,
  DataType.u8: 8,
  DataType.u16: 16,
  DataType.u32: 32,
  DataType.u64: 64,
  DataType.s2: 2,
  DataType.s4: 4,
  DataType.s8: 8,
  DataType.s16: 16,
  DataType.s32: 32,
  DataType.s64: 64,
  DataType.e4m3: 8,
  DataType.e5m2: 8,
  DataType.f8: 8,
  DataType.f6: 6,
  DataType.f4: 4,
  DataType.e2m3: 6,
  DataType.e3m2: 6,
  DataType.e2m1: 4,
  DataType.ue8m0: 8,
  DataType.ue4m3: 8,
  DataType.f16: 16,
  DataType.bf16: 16,
  DataType.f32: 32,
  DataType.tf32: 32,
  DataType.f64: 64,
  DataType.cf16: 32,
  DataType.cbf16: 32,
  DataType.cf32: 64,
  DataType.ctf32: 32,
  DataType.cf64: 128,
  DataType.cu2: 4,
  DataType.cu4: 8,
  DataType.cu8: 16,
  DataType.cu16: 32,
  DataType.cu32: 64,
  DataType.cu64: 128,
  DataType.cs2: 4,
  DataType.cs4: 8,
  DataType.cs8: 16,
  DataType.cs16: 32,
  DataType.cs32: 64,
  DataType.cs64: 128,
}

###################################################################################################

#
class MathOperation(enum.Enum):
  multiply_add = enum_auto()

#
MathOperationTag = {
  MathOperation.multiply_add: 'mutlass::arch::OpMultiplyAdd',
}

###################################################################################################

#
class LayoutType(enum.Enum):
  ColumnMajor = enum_auto()
  RowMajor = enum_auto()
#
LayoutTag = {
  LayoutType.ColumnMajor: 'mutlass::layout::ColumnMajor',
  LayoutType.RowMajor: 'mutlass::layout::RowMajor',
}

#
TransposedLayout = {
  LayoutType.ColumnMajor: LayoutType.RowMajor,
  LayoutType.RowMajor: LayoutType.ColumnMajor,
}

#
ShortLayoutTypeNames = {
  LayoutType.ColumnMajor: 'n',
  LayoutType.RowMajor: 't',
}

###################################################################################################
class KernelScheduleType(enum.Enum):
  ScheduleAuto = enum_auto()
  Multistage = enum_auto()
  Tme = enum_auto()
  TmeWarpSpecialized = enum_auto()
#
KernelScheduleTag = {
  KernelScheduleType.ScheduleAuto: 'mutlass::gemm::collective::KernelScheduleAuto',
  KernelScheduleType.Multistage: 'mutlass::gemm::KernelMultistage',
  KernelScheduleType.Tme: 'mutlass::gemm::KernelTme',
  KernelScheduleType.TmeWarpSpecialized: 'mutlass::gemm::KernelTmeWarpSpecialized',
}

#
KernelScheduleSuffixes = {
  KernelScheduleType.ScheduleAuto: '',
  KernelScheduleType.Multistage: '_lsu',
  KernelScheduleType.Tme: '_tme',
  KernelScheduleType.TmeWarpSpecialized: '_tmews',
}

class EpilogueScheduleType(enum.Enum):
  ScheduleAuto = enum_auto()
  EpilogueTransposed = enum_auto()
  WithTme = enum_auto()
  NoSmem = enum_auto()

#
EpilogueScheduleTag = {
  EpilogueScheduleType.ScheduleAuto: 'mutlass::epilogue::collective::EpilogueScheduleAuto',
  EpilogueScheduleType.EpilogueTransposed: 'mutlass::gemm::EpilogueTransposed',
  EpilogueScheduleType.WithTme: 'mutlass::epilogue::WithTme',
  EpilogueScheduleType.NoSmem: 'mutlass::epilogue::NoSmem'
}

#
EpilogueScheduleSuffixes = {
  EpilogueScheduleType.ScheduleAuto: '',
  EpilogueScheduleType.EpilogueTransposed: '',
  EpilogueScheduleType.WithTme: '_tme',
  EpilogueScheduleType.NoSmem: '_epi_nosmem',
}

class TileSchedulerType(enum.Enum):
  Default = enum_auto()
  Persistent = enum_auto()
  StreamK = enum_auto()
#
TileSchedulerTag = {
  TileSchedulerType.Default: 'void',
  TileSchedulerType.Persistent: 'mutlass::gemm::PersistentScheduler',
  TileSchedulerType.StreamK: 'mutlass::gemm::StreamKScheduler',
}

#
TileSchedulerSuffixes = {
  TileSchedulerType.Default: '',
  TileSchedulerType.Persistent: '',
  TileSchedulerType.StreamK: '_stream_k',
}

###################################################################################################

#
class SideMode(enum.Enum):
  Left = enum_auto()
  Right = enum_auto()

#
SideModeTag = {
  SideMode.Left: 'mutlass::SideMode::kLeft',
  SideMode.Right: 'mutlass::SideMode::kRight'
}

#
ShortSideModeNames = {
  SideMode.Left: 'ls',
  SideMode.Right: 'rs'
}

###################################################################################################

#
class FillMode(enum.Enum):
  Lower = enum_auto()
  Upper = enum_auto()

#
FillModeTag = {
  FillMode.Lower: 'mutlass::FillMode::kLower',
  FillMode.Upper: 'mutlass::FillMode::kUpper'
}

#
ShortFillModeNames = {
  FillMode.Lower: 'l',
  FillMode.Upper: 'u'
}

###################################################################################################

#
class DiagType(enum.Enum):
  NonUnit = enum_auto()
  Unit = enum_auto()

#
DiagTypeTag = {
  DiagType.NonUnit: 'mutlass::DiagType::kNonUnit',
  DiagType.Unit: 'mutlass::DiagType::kUnit'
}

#
ShortDiagTypeNames = {
  DiagType.NonUnit: 'nu',
  DiagType.Unit: 'un'
}

###################################################################################################

#
class OpcodeClass(enum.Enum):
  Simt = enum_auto()
  TensorOp = enum_auto()

OpcodeClassNames = {
  OpcodeClass.Simt: 'simt',
  OpcodeClass.TensorOp: 'tensorop',
}

OpcodeClassTag = {
  OpcodeClass.Simt: 'mutlass::arch::OpClassSimt',
  OpcodeClass.TensorOp: 'mutlass::arch::OpClassTensorOp',
}

###################################################################################################

#
class OperationKind(enum.Enum):
  Gemm = enum_auto()

#
OperationKindNames = {
  OperationKind.Gemm: 'gemm'
}

#
class Target(enum.Enum):
  library = enum_auto()
#
ArchitectureNames = {
  22: 'mp22',
}

#
SharedMemPerCC = {
  22:  72,
}

###################################################################################################

#
def SubstituteTemplate(template, values):
  text = template
  changed = True
  while changed:
    changed = False
    for key, value in values.items():
      regex = "\\$\\{%s\\}" % key
      newtext = re.sub(regex, value, text)
      if newtext != text:
        changed = True
      text = newtext
  return text

###################################################################################################

#
class GemmKind(enum.Enum):
  Universal3x = enum_auto()
  Grouped = enum_auto()
#
GemmKindNames = {
  GemmKind.Universal3x: "gemm",
  GemmKind.Grouped: "gemm_grouped",
}

#
class EpilogueFunctor(enum.Enum):
  LinearCombination = enum_auto()
  LinearCombinationClamp = enum_auto()

#
EpilogueFunctorTag = {
  EpilogueFunctor.LinearCombination: 'mutlass::epilogue::thread::LinearCombination',
  EpilogueFunctor.LinearCombinationClamp: 'mutlass::epilogue::thread::LinearCombinationClamp',
}

#
class SwizzlingFunctor(enum.Enum):
  Identity1 = enum_auto()
  Identity2 = enum_auto()
  Identity4 = enum_auto()
  Identity8 = enum_auto()
  Horizontal = enum_auto()
  StridedDgradIdentity1 = enum_auto()
  StridedDgradIdentity4 = enum_auto()
  StridedDgradHorizontal = enum_auto()
  StreamK = enum_auto()

#
SwizzlingFunctorTag = {
  SwizzlingFunctor.Identity1: 'mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>',
  SwizzlingFunctor.Identity2: 'mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<2>',
  SwizzlingFunctor.Identity4: 'mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<4>',
  SwizzlingFunctor.Identity8: 'mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>',
  SwizzlingFunctor.Horizontal: 'mutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle',
  SwizzlingFunctor.StridedDgradIdentity1: 'mutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<1>',
  SwizzlingFunctor.StridedDgradIdentity4: 'mutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<4>',
  SwizzlingFunctor.StridedDgradHorizontal: 'mutlass::conv::threadblock::StridedDgradHorizontalThreadblockSwizzle',
  SwizzlingFunctor.StreamK: 'mutlass::gemm::threadblock::ThreadblockSwizzleStreamK',
}

#
class GroupScheduleMode(enum.Enum):
  Device = enum_auto(),
  Host = enum_auto()

#
GroupScheduleModeTag = {
  GroupScheduleMode.Device: 'mutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly',
  GroupScheduleMode.Host: 'mutlass::gemm::kernel::GroupScheduleMode::kHostPrecompute'
}

#
ShortGroupScheduleModeNames = {
  GroupScheduleMode.Device: 'Device',
  GroupScheduleMode.Host: 'Host'
}

###################################################################################################

#
class IteratorAlgorithm(enum.Enum):
  Analytic = 0
  Optimized = 1
  FixedChannels = 2
  FewChannels = 3
  FixedStrideDilation = 4

#
IteratorAlgorithmTag = {
  IteratorAlgorithm.Analytic: 'mutlass::conv::IteratorAlgorithm::kAnalytic',
  IteratorAlgorithm.Optimized: 'mutlass::conv::IteratorAlgorithm::kOptimized',
  IteratorAlgorithm.FixedChannels: 'mutlass::conv::IteratorAlgorithm::kFixedChannels',
  IteratorAlgorithm.FewChannels: 'mutlass::conv::IteratorAlgorithm::kFewChannels',
  IteratorAlgorithm.FixedStrideDilation: 'mutlass::conv::IteratorAlgorithm::kFixedStrideDilation'
}

IteratorAlgorithmNames = {
  IteratorAlgorithm.Analytic: 'analytic',
  IteratorAlgorithm.Optimized: 'optimized',
  IteratorAlgorithm.FixedChannels: 'fixed_channels',
  IteratorAlgorithm.FewChannels: 'few_channels',
  IteratorAlgorithm.FixedStrideDilation: 'fixed_stride_dilation'
}

#
class StrideSupport(enum.Enum):
  Strided = 0
  Unity = 1
  Fixed = 2

#
StrideSupportTag = {
  StrideSupport.Strided: 'mutlass::conv::StrideSupport::kStrided',
  StrideSupport.Unity: 'mutlass::conv::StrideSupport::kUnity',
  StrideSupport.Fixed: 'mutlass::conv::StrideSupport::kFixed'
}

StrideSupportNames = {
  StrideSupport.Strided: '',
  StrideSupport.Unity: 'unity_stride',
  StrideSupport.Fixed: 'fixed_stride'
}

#
class GroupMode(enum.Enum):
  NoneGroup = enum_auto()         # dense conv (G=1)
  SingleGroup = enum_auto()       # grouped convolution (single group per CTA)
  MultipleGroup = enum_auto()     # grouped convolution ( multiple groups per CTA)
  Depthwise = enum_auto()         # Depthwise convolution ( C=K=G )

#
GroupModeTag = {
  GroupMode.NoneGroup: 'mutlass::conv::GroupMode::kNone',
  GroupMode.SingleGroup: 'mutlass::conv::GroupMode::kSingleGroup',
  GroupMode.MultipleGroup: 'mutlass::conv::GroupMode::kMultipleGroup',
  GroupMode.Depthwise: 'mutlass::conv::GroupMode::kDepthwise',
}

GroupModeNames = {
  GroupMode.NoneGroup: '',
  GroupMode.SingleGroup: 'single_group',
  GroupMode.MultipleGroup: 'multiple_group',
  GroupMode.Depthwise: 'depthwise',
}

###################################################################################################

#
class MathInstruction:
  def __init__(self,
      instruction_shape,                                            \
      element_a, element_b, element_accumulator,                    \
      opcode_class, math_operation = MathOperation.multiply_add     \
    ):

    self.instruction_shape = instruction_shape
    self.element_a = element_a
    self.element_b = element_b
    self.element_accumulator = element_accumulator
    self.opcode_class = opcode_class
    self.math_operation = math_operation
#
class TileDescription:

  def __init__(self, threadblock_shape, stages, math_instruction, min_compute, max_compute, cluster_shape = [1,1,1]):
    self.threadblock_shape = threadblock_shape
    self.tile_shape = threadblock_shape
    self.stages = stages
    # self.warp_count = warp_count
    self.math_instruction = copy.deepcopy(math_instruction)
    self.minimum_compute_capability = min_compute
    self.maximum_compute_capability = max_compute
    self.cluster_shape = cluster_shape

  def procedural_name(self):
    if self.minimum_compute_capability >= 90:
      return "{tbm}x{tbn}x{tbk}_{cm}x{cn}x{ck}_{s}".format(
        tbm = self.threadblock_shape[0],
        tbn = self.threadblock_shape[1],
        tbk = self.threadblock_shape[2],
        cm = self.cluster_shape[0],
        cn = self.cluster_shape[1],
        ck = self.cluster_shape[2],
        s = self.stages)
    else:
      return "%dx%dx%d_%d" % (self.threadblock_shape[0], self.threadblock_shape[1], self.threadblock_shape[2], self.stages)

#
class TensorDescription:
  def __init__(self, element, layout, alignment = 1 ):
    self.element = element
    self.layout = layout
    self.alignment = alignment
#
def CalculateSmemUsage(operation):
  cta_shape = operation.tile_description.threadblock_shape
  stages = operation.tile_description.stages

  # Few BLAS3 operations only have A tensor
  data_type_size_a = DataTypeSize[operation.A.element]
  data_type_size_b = DataTypeSize[operation.A.element]
  if operation.is_mixed_input():
    data_type_size_b = DataTypeSize[operation.B.element]

  smem_per_stage = data_type_size_a * cta_shape[0] * cta_shape[2] // 8 + \
                    data_type_size_b * cta_shape[1] * cta_shape[2] // 8

  smem_usage = smem_per_stage * stages
  return (smem_usage >> 10)


class LayoutToString:
  def __init__(self, layout):
    self.is_underscore = False
    self.stride = None
    if isinstance(layout, int):
      self.shape = [layout]
    elif len(layout) == 2 and isinstance(layout[0], list) and isinstance(layout[1], list):
      self.shape, self.stride = layout
    elif len(layout) == 1 and isinstance(layout[0], list):
      self.shape = layout[0]
    elif len(layout) == 1 and isinstance(layout[0], Underscore):
      self.is_underscore = True
    else:
      assert False, "format of layout is not supported!"
  def __str__(self):
    if self.is_underscore:
      return "mute::Underscore"
    if self.stride == None:
      shape = self._format(self.shape, "Shape")
      return f"mute::Layout<{shape}>"
    else:
      shape = self._format(self.shape, "Shape")
      stride = self._format(self.stride, "Stride")
      return f"mute::Layout<{shape},{stride}>"

  def _format(self, input, prefix):
    if isinstance(input, list):
      return f"mute::{prefix}<{', '.join(map(lambda x: self._format(x,prefix), input))}>"
    else:
      return f"mute::Int<{input}>"

class CompilerOptions:
    def __init__(self):
        self.options = {}

    def update(self, option):
        mcc_pass, flag = option.split("=")
        if mcc_pass in self.options:
            logging.warning(f"Pass {mcc_pass} is already set. Update it to {flag}")
        self.options[mcc_pass] = flag
        return self

    def str(self):
        output = ""
        for key, value in self.options.items():
            output += f"-mllvm {key}={value} "
        return output.rstrip()

    def __repr__(self):
        return self.str()
