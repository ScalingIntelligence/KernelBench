# Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

set(TEST_NON_CAUSAL_00 --q=433 --k=433 --b=3 --h=4 --h_k=2 --verify)
set(TEST_NON_CAUSAL_01 --q=333 --k=433 --b=3 --h=4 --h_k=2 --verify)
set(TEST_NON_CAUSAL_02 --q=333 --k=599 --b=3 --h=4 --h_k=2 --verify)
set(TEST_NON_CAUSAL_03 --q=599 --k=333 --b=3 --h=4 --h_k=2 --verify)

set(TEST_CAUSAL_00 --q=433 --k=433 --b=3 --h=4 --h_k=2 --causal --verify)
set(TEST_CAUSAL_01 --q=333 --k=433 --b=3 --h=4 --h_k=2 --causal --verify)
set(TEST_CAUSAL_02 --q=333 --k=599 --b=3 --h=4 --h_k=2 --causal --verify)
set(TEST_CAUSAL_03 --q=599 --k=333 --b=3 --h=4 --h_k=2 --causal --verify)

#TODO: add real varlen test cases...
set(TEST_VARLEN_NON_CAUSAL_00 --q=433 --k=433 --b=3 --h=4 --h_k=2 --varlen --verify)
set(TEST_VARLEN_NON_CAUSAL_01 --q=333 --k=433 --b=3 --h=4 --h_k=2 --varlen --verify)
set(TEST_VARLEN_NON_CAUSAL_02 --q=333 --k=599 --b=3 --h=4 --h_k=2 --varlen --verify)
set(TEST_VARLEN_NON_CAUSAL_03 --q=599 --k=333 --b=3 --h=4 --h_k=2 --varlen --verify)

set(TEST_VARLEN_CAUSAL_00 --q=433 --k=433 --b=3 --h=4 --h_k=2 --varlen --causal --verify)
set(TEST_VARLEN_CAUSAL_01 --q=333 --k=433 --b=3 --h=4 --h_k=2 --varlen --causal --verify)
set(TEST_VARLEN_CAUSAL_02 --q=333 --k=599 --b=3 --h=4 --h_k=2 --varlen --causal --verify)
set(TEST_VARLEN_CAUSAL_03 --q=599 --k=333 --b=3 --h=4 --h_k=2 --varlen --causal --verify)

set(TEST_MLA_00 --k=1233 --b=2 --tp=8 --verify)
set(TEST_MLA_01 --k=1233 --b=2 --tp=4 --verify)
set(TEST_MLA_02 --k=1233 --b=2 --tp=1 --verify)

set(TEST_MLA_MTP_00 --k=1233 --b=2 --tp=8 --q=2 --verify)
set(TEST_MLA_MTP_01 --k=1233 --b=2 --tp=4 --q=2 --verify)
set(TEST_MLA_MTP_02 --k=1233 --b=2 --tp=1 --q=2 --verify)

set(TEST_PAGED_NON_CAUSAL_00 --q=433 --k=433 --b=3 --h=4 --h_k=2 --verify)
set(TEST_PAGED_NON_CAUSAL_01 --q=333 --k=433 --b=3 --h=4 --h_k=2 --verify)
set(TEST_PAGED_NON_CAUSAL_02 --q=333 --k=599 --b=3 --h=4 --h_k=2 --verify)
set(TEST_PAGED_NON_CAUSAL_03 --q=599 --k=333 --b=3 --h=4 --h_k=2 --verify)

set(TEST_PAGED_CAUSAL_00 --q=433 --k=433 --b=3 --h=4 --h_k=2 --causal --verify)
set(TEST_PAGED_CAUSAL_01 --q=333 --k=433 --b=3 --h=4 --h_k=2 --causal --verify)
set(TEST_PAGED_CAUSAL_02 --q=333 --k=599 --b=3 --h=4 --h_k=2 --causal --verify)
set(TEST_PAGED_CAUSAL_03 --q=599 --k=333 --b=3 --h=4 --h_k=2 --causal --verify)


function(add_tests OUT_LIST)
  set(result "")
  foreach(test_name IN LISTS ARGN)
    set(${test_name}_128_128 "${${test_name}} --d_qk=128 --d_vo=128" PARENT_SCOPE)
    list(APPEND result ${test_name}_128_128)
    set(${test_name}_192_128 "${${test_name}} --d_qk=192 --d_vo=128" PARENT_SCOPE)
    list(APPEND result ${test_name}_192_128)
  endforeach()
  set(${OUT_LIST} "${result}" PARENT_SCOPE)
endfunction()

set(TEST_NON_CAUSAL_00 "--q=433 --k=433 --b=3 --h=4 --h_k=2 --verify")
set(TEST_NON_CAUSAL_01 "--q=333 --k=433 --b=3 --h=4 --h_k=2 --verify")
set(TEST_NON_CAUSAL_02 "--q=333 --k=599 --b=3 --h=4 --h_k=2 --verify")
set(TEST_NON_CAUSAL_03 "--q=599 --k=333 --b=3 --h=4 --h_k=2 --verify")

set(TEST_CAUSAL_00 "--q=433 --k=433 --b=3 --h=4 --h_k=2 --causal --verify")
set(TEST_CAUSAL_01 "--q=333 --k=433 --b=3 --h=4 --h_k=2 --causal --verify")
set(TEST_CAUSAL_02 "--q=333 --k=599 --b=3 --h=4 --h_k=2 --causal --verify")
set(TEST_CAUSAL_03 "--q=599 --k=333 --b=3 --h=4 --h_k=2 --causal --verify")

set(TEST_VARLEN_NON_CAUSAL_00 "--q=433 --k=433 --b=3 --h=4 --h_k=2 --varlen --verify")
set(TEST_VARLEN_NON_CAUSAL_01 "--q=333 --k=433 --b=3 --h=4 --h_k=2 --varlen --verify")
set(TEST_VARLEN_NON_CAUSAL_02 "--q=333 --k=599 --b=3 --h=4 --h_k=2 --varlen --verify")
set(TEST_VARLEN_NON_CAUSAL_03 "--q=599 --k=333 --b=3 --h=4 --h_k=2 --varlen --verify")

set(TEST_VARLEN_CAUSAL_00 "--q=433 --k=433 --b=3 --h=4 --h_k=2 --varlen --causal --verify")
set(TEST_VARLEN_CAUSAL_01 "--q=333 --k=433 --b=3 --h=4 --h_k=2 --varlen --causal --verify") # error?
set(TEST_VARLEN_CAUSAL_02 "--q=333 --k=599 --b=3 --h=4 --h_k=2 --varlen --causal --verify")
set(TEST_VARLEN_CAUSAL_03 "--q=599 --k=333 --b=3 --h=4 --h_k=2 --varlen --causal --verify")

set(TEST_MLA_00 "--k=1233 --b=2 --tp=8 --verify")
set(TEST_MLA_01 "--k=1233 --b=2 --tp=4 --verify")
set(TEST_MLA_02 "--k=1233 --b=2 --tp=1 --verify")

set(TEST_MLA_MTP_00 "--k=1233 --b=2 --tp=8 --q=2 --verify")
set(TEST_MLA_MTP_01 "--k=1233 --b=2 --tp=4 --q=2 --verify")
set(TEST_MLA_MTP_02 "--k=1233 --b=2 --tp=1 --q=2 --verify")

add_tests(FMHA_TESTS
  TEST_NON_CAUSAL_00
  TEST_NON_CAUSAL_01
  TEST_NON_CAUSAL_02
  TEST_NON_CAUSAL_03
  TEST_CAUSAL_00
  TEST_CAUSAL_01
  TEST_CAUSAL_02
  TEST_CAUSAL_03
  TEST_VARLEN_NON_CAUSAL_00
  TEST_VARLEN_NON_CAUSAL_01
  TEST_VARLEN_NON_CAUSAL_02
  TEST_VARLEN_NON_CAUSAL_03
  TEST_VARLEN_CAUSAL_00
  TEST_VARLEN_CAUSAL_01
  TEST_VARLEN_CAUSAL_02
  TEST_VARLEN_CAUSAL_03
)

function(mutlass_experimental_mp31_fmha_executable NAME)
  set(options)
  set(oneValueArgs DISABLE_TESTS)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)

  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  # add -fmusa-flush-denormals-to-zero to enable fast exp2 instruction
  list(APPEND MUTLASS_MUSA_MCC_FLAGS -fmusa-flush-denormals-to-zero -ffast-math)

  mutlass_add_executable(${NAME} ${__UNPARSED_ARGUMENTS})

  add_dependencies(mutlass_experimental ${NAME})

  target_link_libraries(
    ${NAME}
    PRIVATE
    MUTLASS
    mutlass_tools_util_includes
    musa_driver
    musart
  )

  target_include_directories(
    ${NAME}
    PRIVATE
    ${MUTLASS_EXPERIMENTAL_COMMON_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  install(
    TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )

  mutlass_add_executable_tests(
    test_examples_${NAME} ${NAME}
    DEPENDS ${__DEPENDS}
    DEPENDEES test_experimental ${__DEPENDEES}
    TEST_COMMAND_OPTIONS ${__TEST_COMMAND_OPTIONS}
    DISABLE_EXECUTABLE_INSTALL_RULE
    DISABLE_TESTS ${__DISABLE_TESTS}
  )
endfunction()

mutlass_experimental_mp31_fmha_executable(
  mp31_fmha_fwd
  fmha_fwd.mu
  TEST_COMMAND_OPTIONS
  ${FMHA_TESTS}
)

mutlass_experimental_mp31_fmha_executable(
  mp31_mla_decode
  mla.mu
  TEST_COMMAND_OPTIONS
  TEST_MLA_00
  TEST_MLA_01
  TEST_MLA_02
  TEST_MLA_MTP_00
  TEST_MLA_MTP_01
  TEST_MLA_MTP_02
)

mutlass_experimental_mp31_fmha_executable(
  mp31_paged_fmha_fwd
  paged_fmha_fwd.mu
  TEST_COMMAND_OPTIONS
  TEST_NON_CAUSAL_00
  TEST_NON_CAUSAL_01
  TEST_NON_CAUSAL_02
  TEST_NON_CAUSAL_03
  TEST_CAUSAL_00
  TEST_CAUSAL_01
  TEST_CAUSAL_02
  TEST_CAUSAL_03
)
