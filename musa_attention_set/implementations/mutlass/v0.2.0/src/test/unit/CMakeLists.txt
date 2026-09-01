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

set(MUTLASS_UNIT_TEST_COMMON_DIR ${CMAKE_CURRENT_LIST_DIR}/common)

mutlass_add_library(
  mutlass_test_unit_infra 
  common/filter_architecture.cpp
  )

target_link_libraries(
  mutlass_test_unit_infra
  PUBLIC
  MUTLASS
  mutlass_tools_util_includes
  GTest::gtest
  )

mutlass_add_library(
  mutlass_test_unit_infra_lib
  test_unit.cpp
  )

target_link_libraries(
  mutlass_test_unit_infra_lib
  PUBLIC
  mutlass_test_unit_infra
  )

set(MUTLASS_TEST_UNIT_RESULTS_CACHE_DIR ${CMAKE_CURRENT_LIST_DIR}/data/hashes)

function(mutlass_test_unit_add_executable NAME)
  set(options WITHOUT_MUSA)
  set(oneValueArgs)
  set(multiValueArgs)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  mutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS})

  target_compile_definitions(${NAME} PUBLIC MUTLASS_TARGET_NAME="${NAME}")

  target_include_directories(
    ${NAME}
    PRIVATE
    ${MUTLASS_UNIT_TEST_COMMON_DIR}
  )

  if (__WITHOUT_MUSA)
    target_link_libraries(
      ${NAME}
      PUBLIC
      GTest::gtest
    )
  else()
    target_link_libraries(
      ${NAME}
      PRIVATE
      mutlass_test_unit_infra
      mutlass_test_unit_infra_lib
      musart
      musa_driver
    )
  endif()

  string(REGEX REPLACE mutlass_ "" NAME_STEM ${NAME})

  set(RESULT_CACHE_FILE "${MUTLASS_TEST_UNIT_RESULTS_CACHE_DIR}/cached_results_${NAME}.txt")

  if (EXISTS ${RESULT_CACHE_FILE})
    set(RESULT_CACHE_FILE_ARGS RESULT_CACHE_FILE ${RESULT_CACHE_FILE})
  endif()

  set(MUTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS --gtest_output=xml:${NAME_STEM}.gtest.xml)

  mutlass_add_executable_tests(
    ${NAME_STEM} ${NAME}
    TEST_COMMAND_OPTIONS MUTLASS_TEST_UNIT_TEST_COMMAND_OPTIONS
    ${RESULT_CACHE_FILE_ARGS}
  )

endfunction()

add_custom_target(mutlass_test_unit)
add_custom_target(test_unit)

set(SUBDIRS
  mute
  gemm
)

foreach(SUBDIR ${SUBDIRS})
  add_subdirectory(${SUBDIR})
  add_dependencies(mutlass_test_unit mutlass_test_unit_${SUBDIR})
  add_dependencies(test_unit test_unit_${SUBDIR})
endforeach()
