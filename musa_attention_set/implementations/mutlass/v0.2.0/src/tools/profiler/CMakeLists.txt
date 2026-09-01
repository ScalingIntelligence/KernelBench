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

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

#
# Sources for MUTLASS Profiler Tool
#
set(MUTLASS_TOOLS_PROFILER_SOURCES
  src/main.cpp
  src/mutlass_profiler.mu
  src/options.mu
  src/performance_report.cpp
  src/enumerated_types.cpp
  src/gpu_timer.cpp
  src/device_allocation.mu
  src/device_context.mu
  src/problem_space.cpp
  src/operation_profiler.mu
  src/gemm_operation_profiler.mu
)

#
# Build target
#

mutlass_add_executable(
  mutlass_profiler
  ${MUTLASS_TOOLS_PROFILER_SOURCES}
)
add_executable(mt::mutlass::profiler ALIAS mutlass_profiler)
set_target_properties(mutlass_profiler PROPERTIES EXPORT_NAME profiler)

#
# Include paths
#
target_include_directories(
  mutlass_profiler
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${PROJECT_BINARY_DIR}/include 
  )

#
# Library dependencies
#

target_link_libraries(
  mutlass_profiler
  PRIVATE 
  mutlass_lib
  mutlass_tools_util_includes
  $<$<BOOL:${MUTLASS_ENABLE_MUBLAS}>:mt::mublas>
  $<$<BOOL:${MUTLASS_ENABLE_MUDNN}>:mt::mudnn>
  musart
  musa_driver
  )

install(
  TARGETS mutlass_profiler
  EXPORT MtMutlass
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  set(MUTLASS_PROFILER_TEST_COMMAND_OPTIONS_GEMM --operation=Gemm --providers=mutlass --junit-output=test_mutlass_profiler_gemm --print-kernel-before-running=true)


  mutlass_add_executable_tests(
    test_profiler mutlass_profiler
    DEPENDEES test_all
    TEST_COMMAND_OPTIONS
      GEMM
    TEST_COMMAND_OPTIONS_PREFIX
      MUTLASS_PROFILER_TEST_COMMAND_OPTIONS_
    DISABLE_EXECUTABLE_INSTALL_RULE
    )
