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
Utilities for emitting GEMM kernels
"""

import collections
import enum
import functools
import operator
import os.path
import shutil

try:
  import builtins
  if hasattr(builtins, "MUTLASS_IGNORE_PACKAGE") and MUTLASS_IGNORE_PACKAGE == True:
    raise ImportError("Disabling attempt to import mutlass_library")
  from mutlass_library.library import *
except ImportError:
  from library import *

###################################################################################################
#
# Data structure modeling a GEMM operation
#
###################################################################################################

#
class GemmOperation:
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, D = None,
      kernel_schedule = KernelScheduleType.ScheduleAuto, epilogue_schedule = EpilogueScheduleType.ScheduleAuto,
      tile_scheduler = TileSchedulerType.Default, extra_args = None):

    kinds_3x = {
      GemmKind.Universal3x,
    }
    self.is_3x = gemm_kind in kinds_3x
    self.prefix = ""
    self.operation_kind = OperationKind.Gemm
    self.arch = arch
    self.tile_description = tile_description
    self.gemm_kind = gemm_kind
    self.A = A
    self.B = B
    self.C = C
    self.D = D

    if self.D == None:
      self.D = self.C

    if not self.is_3x:
      assert(kernel_schedule == KernelScheduleType.ScheduleAuto)
      assert(epilogue_schedule == EpilogueScheduleType.ScheduleAuto)
    self.kernel_schedule = kernel_schedule
    self.epilogue_schedule = epilogue_schedule
    self.element_epilogue = element_epilogue
    self.epilogue_functor = epilogue_functor
    self.swizzling_functor = swizzling_functor
    self.tile_scheduler = tile_scheduler

  #
  def is_mixed_input(self):
    return self.A.element != self.B.element

  #
  def accumulator_type(self):
    accum = self.tile_description.math_instruction.element_accumulator
    return accum

  #
  def short_math_name(self):
    return ShortDataTypeNames[self.accumulator_type()]


  #
  def core_name(self):
    ''' The basic operation kind is prefixed with a letter indicating the accumulation type. '''

    inst_shape = ''
    inst_operation = ''
    intermediate_type = ''

    tensor_ops = [
      OpcodeClass.TensorOp,
    ]

    is_tensor_op = self.tile_description.math_instruction.opcode_class in tensor_ops

    if is_tensor_op:

      math_op = self.tile_description.math_instruction.math_operation
      math_op_string = ''

      if self.is_3x:
        inst_shape = "{0}x{1}x{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape))
      else:
        inst_shape = "{0}{1}{2}".format(*tuple(self.tile_description.math_instruction.instruction_shape))

      inst_shape += math_op_string

      if self.tile_description.math_instruction.element_a != self.A.element and \
        self.tile_description.math_instruction.element_a != self.tile_description.math_instruction.element_accumulator:
        intermediate_type = DataTypeNames[self.tile_description.math_instruction.element_a]

    return "%s%s%s%s" % (self.short_math_name(), inst_shape, intermediate_type, GemmKindNames[self.gemm_kind])

  # Generates a string representing the MMA instruction.
  def extended_name(self):
    ''' Append data types if they differ from compute type. '''
    if self.C.element != self.tile_description.math_instruction.element_accumulator and \
      self.A.element != self.tile_description.math_instruction.element_accumulator:
      extended_name = "${element_c}_${core_name}_${element_a}"
      if self.is_mixed_input():
        extended_name += "_${element_b}"
    elif self.C.element == self.tile_description.math_instruction.element_accumulator and  \
      self.A.element != self.tile_description.math_instruction.element_accumulator:
      extended_name = "${core_name}_${element_a}"
      if self.is_mixed_input():
        extended_name += "_${element_b}"
    else:
      extended_name = "${core_name}"

    extended_name = SubstituteTemplate(extended_name, {
      'element_a': DataTypeNames[self.A.element],
      'element_b': DataTypeNames[self.B.element],
      'element_c': DataTypeNames[self.C.element],
      'core_name': self.core_name()
      })

    return extended_name

  def extended_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    extended_name = "{core_name}_{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element],
      core_name = self.core_name())
    return extended_name

  def datatype_name_3x(self):
    '''Generates a string representing the MMA atom. Assumes accumulator type is C type.'''
    datatype_name = "{element_a}_{element_b}_{element_acc}_{element_c}_{element_d}".format(
      element_a = DataTypeNames[self.A.element],
      element_b = DataTypeNames[self.B.element],
      element_acc = DataTypeNames[self.tile_description.math_instruction.element_accumulator],
      element_c = DataTypeNames[self.C.element],
      element_d = DataTypeNames[self.D.element])
    return datatype_name

  # Generates a short string representing the AB layout tags (e.g. nt or tn)
  def layout_name(self):
    return "%s%s" % (ShortLayoutTypeNames[self.A.layout], ShortLayoutTypeNames[self.B.layout])

  # Generates a short string representing the ABC layout tags (e.g. ntn or tnn)
  def layout_name_3x(self):
    return "{}{}{}".format(
      ShortLayoutTypeNames[self.A.layout],
      ShortLayoutTypeNames[self.B.layout],
      ShortLayoutTypeNames[self.C.layout])

  # Generates a short string representing underlying kernel schedule type
  def kernel_schedule_name_3x(self):
    return KernelScheduleSuffixes[self.kernel_schedule]

  # Generates a short string representing underlying epilogue schedule type
  def epilogue_schedule_name_3x(self):
    return EpilogueScheduleSuffixes[self.epilogue_schedule]

  # Generate a short string representing the operation class
  def opcode_class_name(self):
    return OpcodeClassNames[self.tile_description.math_instruction.opcode_class]

  # Generates the full kernel function name
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    opcode_class_name = OpcodeClassNames[self.tile_description.math_instruction.opcode_class]
    kernel_name_template = "mutlass{p}_mp{ar}_{op}_{ex}_{tbm}x{tbn}x{tbk}_{l}_{s}_align{al}{t}{k}{e}"
    return kernel_name_template.format(
        p = self.prefix,
        ar = self.arch,
        op = opcode_class_name,
        ex = self.extended_name_3x(),
        tbm = self.tile_description.tile_shape[0],
        tbn = self.tile_description.tile_shape[1],
        tbk = self.tile_description.tile_shape[2],
        l = self.tile_description.stages,
        s = self.layout_name_3x(),
        al = str(max(self.A.alignment, self.B.alignment)),
        t = TileSchedulerSuffixes[self.tile_scheduler],
        k = self.kernel_schedule_name_3x(),
        e = self.epilogue_schedule_name_3x()
      )

  #
  def configuration_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    return self.procedural_name()

  def __hash__(self):
    return hash(self.configuration_name())

  def __eq__(self, other):
    return self.configuration_name() == other.configuration_name()

###################################################################################################
#
# Data structure modeling a grouped GEMM operation
#
###################################################################################################

#
class GroupedGemmOperation(GemmOperation):
  #
  def __init__(self, gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
      epilogue_functor = EpilogueFunctor.LinearCombination, swizzling_functor = SwizzlingFunctor.Identity8, \
      scheduler_mode = GroupScheduleMode.Device):
    super().__init__(gemm_kind, arch, tile_description, A, B, C, element_epilogue, \
                     epilogue_functor, swizzling_functor)

    self.scheduler_mode = scheduler_mode

  #
  def procedural_name(self):
    ''' The full procedural name indicates architecture, extended name, tile size, and layout. '''
    base = super().procedural_name()
    return SubstituteTemplate(
      base + "_schedule${schedule}",
      {
        'schedule': ShortGroupScheduleModeNames[self.scheduler_mode]
      })


###################################################################################################
#
# Emits single instances of a MUTLASS device-wide operator
#
###################################################################################################
class EmitGemmUniversal3xInstance:
  ''' Responsible for emitting a MUTLASS 3.x template definition'''

  def __init__(self, operation_suffix = ''):
    self.operation_suffix = operation_suffix
    self.includes = [
      "mutlass/mutlass.h",
      "mutlass/gemm/device/gemm_universal_adapter.h",
      "mutlass/gemm/collective/collective_builder.hpp",
      "mutlass/epilogue/collective/collective_builder.hpp",
    ]
    self.builtin_epilogue_functor_template = """
    ${epilogue_functor}<
      ${element_c},
      ${epilogue_vector_length},
      ${element_accumulator},
      ${element_epilogue}
    >
"""
    self.gemm_template = """

using ${operation_name}_mainloop =
  typename mutlass::gemm::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_main},
    ${element_a}, ${layout_a}, ${align_a},
    ${element_b}, ${layout_b}, ${align_b},
    ${element_accumulator},
    mute::Shape<mute::_${tile_shape_m}, mute::_${tile_shape_n}, mute::_${tile_shape_k}>,
    mute::Shape<mute::_${cluster_m},mute::_${cluster_n},mute::_${cluster_k}>,
    ${stages},
    ${kernel_schedule}
  >::CollectiveOp;

using ThreadEpilogueOp = mutlass::epilogue::fusion::LinearCombination<${element_d},${element_accumulator},${element_c},${element_accumulator}>;

using ${operation_name}_epilogue =
  typename mutlass::epilogue::collective::CollectiveBuilder<
    ${arch}, ${opcode_class_epi},
    mute::Shape<mute::_${tile_shape_m}, mute::_${tile_shape_n}, mute::_${tile_shape_k}>,
    mute::Shape<mute::_${cluster_m},mute::_${cluster_n},mute::_${cluster_k}>,
    ${epi_tile_mn},
    ${element_accumulator}, ${element_epilogue},
    ${element_c}, ${layout_c}, ${align_c},
    ${element_d}, ${layout_d}, ${align_d},
    ${epilogue_schedule},
    ThreadEpilogueOp,
    ${operation_name}_mainloop
  >::CollectiveOp;

// Gemm operator ${operation_name}
using ${operation_name}_base = mutlass::gemm::kernel::GemmUniversal<
    mute::Shape<int,int,int,int>,
    ${operation_name}_mainloop,
    ${operation_name}_epilogue,
    ${tile_scheduler}>;

// Define named type
struct ${operation_name} :
  public ${operation_name}_base { };

"""
  #
  def instance_template(self):
    return """
${compile_guard_start}
  using GemmKernel = mutlass::gemm::device::GemmUniversalAdapter<${operation_name}>;
  manifest.append(
    new ${gemm_kind}<GemmKernel>("${operation_name}"));
${compile_guard_end}
"""

  #
  def emit(self, operation):

    tile_shape = operation.tile_description.tile_shape
    # stage count set to zero indicates builder automatic stage selection
    if operation.tile_description.stages > 0:
      stage_count_string = f"mutlass::gemm::collective::StageCount<{str(operation.tile_description.stages)}>"
    else:
      stage_count_string = f"mutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename {str(operation.procedural_name())}_epilogue::SharedStorage)>"

    epi_tile_mn = "mutlass::epilogue::collective::EpilogueTileAuto"
    opcode_class_main = operation.tile_description.math_instruction.opcode_class
    opcode_class_epi = opcode_class_main

    instance_layout_A, instance_layout_B, instance_layout_C , instance_layout_D = \
      (operation.A.layout, operation.B.layout, operation.C.layout, operation.D.layout)

    # 3.0 profiler integration only supports trivial epilogues for now
    epilogue_vector_length = 1

    # Support built-in epilogue functors or user-defined functions
    if isinstance(operation.epilogue_functor, enum.Enum):
      values = {
        'epilogue_vector_length': str(epilogue_vector_length),
        'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
        'epilogue_functor': EpilogueFunctorTag[operation.epilogue_functor],
      }
      epilogue_functor = SubstituteTemplate(self.builtin_epilogue_functor_template, values)
    else:
      epilogue_functor = self.epilogue_functor.emit_declaration()
    #
    element_a = DataTypeTag[operation.A.element]
    element_b = DataTypeTag[operation.B.element]
    epilogue_schedule_type = EpilogueScheduleTag[operation.epilogue_schedule]
    values = {
      'operation_name': operation.procedural_name(),
      'operation_suffix': self.operation_suffix,
      'element_a': element_a,
      'layout_a': LayoutTag[instance_layout_A],
      'element_b': element_b,
      'layout_b': LayoutTag[instance_layout_B],
      'element_c': DataTypeTag[operation.C.element],
      'layout_c': LayoutTag[instance_layout_C],
      'element_d': DataTypeTag[operation.D.element],
      'layout_d': LayoutTag[instance_layout_D],
      'element_accumulator': DataTypeTag[operation.accumulator_type()],
      'opcode_class_main': OpcodeClassTag[opcode_class_main],
      'opcode_class_epi': OpcodeClassTag[opcode_class_epi],
      'arch': "mutlass::arch::Mp%d" % operation.arch,
      'tile_shape_m': str(operation.tile_description.tile_shape[0]),
      'tile_shape_n': str(operation.tile_description.tile_shape[1]),
      'tile_shape_k': str(operation.tile_description.tile_shape[2]),
      'cluster_m': str(operation.tile_description.cluster_shape[0]),
      'cluster_n': str(operation.tile_description.cluster_shape[1]),
      'cluster_k': str(operation.tile_description.cluster_shape[2]),
      'instruction_shape_m': str(operation.tile_description.math_instruction.instruction_shape[0]),
      'instruction_shape_n': str(operation.tile_description.math_instruction.instruction_shape[1]),
      'instruction_shape_k': str(operation.tile_description.math_instruction.instruction_shape[2]),
      'kernel_schedule' : str(KernelScheduleTag[operation.kernel_schedule]),
      'epilogue_schedule' : str(epilogue_schedule_type),
      'epi_tile_mn' : epi_tile_mn,
      'epilogue_functor': epilogue_functor,
      'stages': stage_count_string,
      'align_a': str(operation.A.alignment),
      'align_b': str(operation.B.alignment),
      'align_c': str(operation.C.alignment),
      'align_d': str(operation.C.alignment),
      'math_operation': MathOperationTag[operation.tile_description.math_instruction.math_operation],
      'epilogue_vector_length': str(epilogue_vector_length),
      'element_epilogue': str(DataTypeTag[operation.element_epilogue]),
      'tile_scheduler': str(TileSchedulerTag[operation.tile_scheduler]),
    }

    return SubstituteTemplate(self.gemm_template, values)

###################################################################################################

#
class EmitGemmConfigurationLibrary:
  def __init__(self, operation_path, configuration_name):
    self.configuration_name = configuration_name
    self.configuration_path = os.path.join(operation_path, "%s.mu" % configuration_name).replace('\\', '/')

    self.instance_emitter = {
      GemmKind.Universal3x: EmitGemmUniversal3xInstance,
    }

    self.gemm_kind_wrappers = {
      GemmKind.Universal3x: 'GemmUniversal3xOperation',
    }

    self.wmma_guard_start = "#if defined(MUTLASS_ARCH_WMMA_SM${sm_number}_ENABLED)"

    self.separator = """
///////////////////////////////////////////////////////////////////////////////////////////////////

"""

    self.header_template = """
/*
  Generated by gemm_operation.py - Do not edit.
*/
"""

    self.initialize_function_template = """

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace library {

///////////////////////////////////////////////////////////////////////////////////////////////////

void initialize_${configuration_name}(Manifest &manifest) {

"""
    self.epilogue_template = """

}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace library
} // namespace mutlass

///////////////////////////////////////////////////////////////////////////////////////////////////

"""

  def __enter__(self):
    self.configuration_file = open(self.configuration_path, "w")
    self.configuration_file.write(self.header_template)
    self.configuration_file.write(self.separator)

    self.includes = collections.OrderedDict([
      ("mutlass/mutlass.h", None),
      ("mutlass/library/library.h", None),
      ("mutlass/library/manifest.h", None),
      ("library_internal.h", None),
      ("gemm_operation_3x.hpp", None),
    ])
    self.instance_definitions = []
    self.instance_wrappers = []

    self.operations = []
    return self

  def emit(self, operation):
    emitter = self.instance_emitter[operation.gemm_kind]()

    for incl in emitter.includes:
      self.includes[incl] = None

    self.operations.append(operation)

    self.instance_definitions.append(emitter.emit(operation))

    self.instance_wrappers.append(SubstituteTemplate(emitter.instance_template(), {
      'configuration_name': self.configuration_name,
      'operation_name': operation.procedural_name(),
      'gemm_kind': self.gemm_kind_wrappers[operation.gemm_kind],
      'compile_guard_start': "",
      'compile_guard_end': ""
      }))

  def __exit__(self, exception_type, exception_value, traceback):

    # Write includes
    for incl, _ in self.includes.items():
      include_statement = "#include \"%s\"\n" % incl
      self.configuration_file.write(include_statement)

    self.configuration_file.write(self.separator)

    # Write instance definitions in top-level namespace
    for instance_definition in self.instance_definitions:
      self.configuration_file.write(instance_definition)

    # Add wrapper objects within initialize() function
    self.configuration_file.write(SubstituteTemplate(self.initialize_function_template, {
      'configuration_name': self.configuration_name
      }))

    for instance_wrapper in self.instance_wrappers:
      self.configuration_file.write(instance_wrapper)

    self.configuration_file.write(self.epilogue_template)
    self.configuration_file.close()

###################################################################################################
###################################################################################################

