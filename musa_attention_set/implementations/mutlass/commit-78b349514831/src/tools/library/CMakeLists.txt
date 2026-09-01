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

# cmake_policy(SET CMP0112 NEW)

include(GNUInstallDirs)

################################################################################

set(MUTLASS_BUILD_MONO_LIBRARY OFF CACHE BOOL
  "Determines whether the mutlass library is generated as a single file or multiple files.")

################################################################################

add_library(mutlass_library_includes INTERFACE)
add_library(mt::mutlass::library::includes ALIAS mutlass_library_includes)
set_target_properties(mutlass_library_includes PROPERTIES EXPORT_NAME library::includes)

target_include_directories(
  mutlass_library_includes
  INTERFACE
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )

target_link_libraries(
  mutlass_library_includes
  INTERFACE
  MUTLASS
  mutlass_tools_util_includes
  )

install(
  TARGETS mutlass_library_includes
  EXPORT MtMutlass
  )

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/
  )

add_library(mutlass_library_internal_interface INTERFACE)
add_library(mt::mutlass::library::obj_interface ALIAS mutlass_library_internal_interface)

target_include_directories(
  mutlass_library_internal_interface
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  )

target_link_libraries(
  mutlass_library_internal_interface
  INTERFACE
  mutlass_library_includes
  )

################################################################################

function(mutlass_add_mutlass_library)
#
# Generates static and shared libraries with the given SOURCES. The public CMake
# targets produces will be mutlass_library(_${SUFFIX})? and
# mutlass_library(_${SUFFIX})?_static.
#
# SUFFIX: An additional string to be joined to the default names. If suffix is given,
#   the generated libraries will be linked as a dependency of the main mutlass library.

  set(options)
  set(oneValueArgs SUFFIX)
  set(multiValueArgs MP_ARCHS MCC_FLAGS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  set(DEFAULT_NAME mutlass_library)

  set(__NAME ${DEFAULT_NAME})
  set(__OUTPUT_NAME mutlass)
  set(__EXPORT_NAME library)

  if (__SUFFIX)
    string(APPEND __NAME _${__SUFFIX})
    string(APPEND __OUTPUT_NAME _${__SUFFIX})
    string(APPEND __EXPORT_NAME _${__SUFFIX})
  endif()
  mutlass_add_library(
    ${__NAME}_objs
    ${__UNPARSED_ARGUMENTS}
    MP_ARCHS_ ${__MP_ARCHS}
    MCC_FLAGS_ ${__MCC_FLAGS}
    )

  target_link_libraries(${__NAME}_objs
    PUBLIC mutlass_library_includes
    PRIVATE mutlass_library_internal_interface
    )

  if (MUTLASS_BUILD_MONO_LIBRARY AND __SUFFIX)

    # If we're only building a single monolithic library then we
    # simply link the generated object files to the default library.

    target_link_libraries(${DEFAULT_NAME} PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>)
    target_link_libraries(${DEFAULT_NAME}_static PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>)

  else()

    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${__NAME}_temp.mu)
    mutlass_add_library(
      ${__NAME}
      SHARED
      EXPORT_NAME ${__EXPORT_NAME}
      ${CMAKE_CURRENT_BINARY_DIR}/${__NAME}_temp.mu
      )

    set_target_properties(
      ${__NAME}
      PROPERTIES
      OUTPUT_NAME ${__OUTPUT_NAME}
      WINDOWS_EXPORT_ALL_SYMBOLS 1
      )

    target_link_libraries(
      ${__NAME}
      PUBLIC mutlass_library_includes
      PRIVATE -Wl,--whole-archive $<BUILD_INTERFACE:${__NAME}_objs>
      -Wl,--no-whole-archive
      musa_driver
      )

    set_target_properties(${__NAME} PROPERTIES DEBUG_POSTFIX "${MUTLASS_LIBRARY_DEBUG_POSTFIX}")

    mutlass_add_library(
      ${__NAME}_static
      STATIC
      EXPORT_NAME ${__EXPORT_NAME}_static
      ${CMAKE_CURRENT_BINARY_DIR}/${__NAME}_temp.mu
      )

    if (WIN32)
      set(STATIC_OUTPUT_NAME ${__OUTPUT_NAME}.static)
    else()
      set(STATIC_OUTPUT_NAME ${__OUTPUT_NAME})
    endif()

    set_target_properties(
      ${__NAME}_static
      PROPERTIES
      OUTPUT_NAME ${STATIC_OUTPUT_NAME}
      WINDOWS_EXPORT_ALL_SYMBOLS 1
      )

    target_link_libraries(
      ${__NAME}_static
      PUBLIC mutlass_library_includes
      PRIVATE $<BUILD_INTERFACE:${__NAME}_objs>
      musa_driver
      )

    set_target_properties(${__NAME}_static PROPERTIES DEBUG_POSTFIX "${MUTLASS_LIBRARY_DEBUG_POSTFIX}")

    install(
      TARGETS ${__NAME} ${__NAME}_static
      EXPORT MtMutlass
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )

    if (__SUFFIX)

      # The partial libraries generated will be registered as linked libraries
      # to the main mutlass library so users automatically get the necessary link
      # commands to pull in all kernels by default.

      target_link_libraries(${DEFAULT_NAME} INTERFACE ${__NAME})
      target_link_libraries(${DEFAULT_NAME}_static INTERFACE ${__NAME}_static)

    endif()

  endif()

endfunction()

################################################################################

mutlass_add_mutlass_library(

  src/handle.mu
  src/manifest.cpp
  src/operation_table.mu
  src/singleton.mu
  src/util.mu

  # files split for parallel compilation
  src/reference/gemm_s8_s8_s32.mu
  src/reference/gemm_u8_u8_s32.mu
  src/reference/gemm_fp32out.mu
  src/reference/gemm_fp_other.mu
  src/reference/initialize_reference_operations.mu
)

# For backward compatibility with the old name
add_library(mutlass_lib ALIAS mutlass_library)

################################################################################

file(GLOB_RECURSE GENERATOR_PYTHON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)

#
# auto-instantiation of MUTLASS kernels
#

# set mutlass generator compiler version to filter kernels in the generator not supported by a specific toolkit.
set(MUTLASS_GENERATOR_MUSA_COMPILER_VERSION ${CMAKE_MUSA_COMPILER_VERSION})
set(MUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated_kernels.txt CACHE STRING "Generated kernel listing file")

# --log-level is set to DEBUG to enable printing information about which kernels were excluded
# from generation in /python/mutlass_library/manifest.py. To avoid having this information appear
# in ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log, set this parameter to INFO
execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../python/mutlass_library
  COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${MUTLASS_LIBRARY_PACKAGE_DIR}
    ${Python3_EXECUTABLE} ${MUTLASS_SOURCE_DIR}/python/mutlass_library/generator.py
    --operations "${MUTLASS_LIBRARY_OPERATIONS}"
    --build-dir ${PROJECT_BINARY_DIR}
    --curr-build-dir ${CMAKE_CURRENT_BINARY_DIR}
    --generator-target library
    --architectures "${MUTLASS_MCC_ARCHS_ENABLED}"
    --kernels "${MUTLASS_LIBRARY_KERNELS}"
    --ignore-kernels "${MUTLASS_LIBRARY_IGNORE_KERNELS}"
    --kernel-filter-file "${MUTLASS_KERNEL_FILTER_FILE}"
    --selected-kernel-list "${MUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}"
    --musa-version "${MUTLASS_GENERATOR_MUSA_COMPILER_VERSION}"
    --log-level DEBUG
    --disable-mutlass-package-imports
  RESULT_VARIABLE mutlass_lib_INSTANCE_GENERATION_RESULT
  OUTPUT_VARIABLE mutlass_lib_INSTANCE_GENERATION_OUTPUT
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
)

if(NOT mutlass_lib_INSTANCE_GENERATION_RESULT EQUAL 0)
  message(FATAL_ERROR "Error generating library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log")
endif()

message(STATUS "Completed generation of library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log for more information.")

# include auto-instantiated kernels in he MUTLASS Deliverables Library
set(MUTLASS_LIBRARY_MANIFEST_CMAKE_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/manifest.cmake)
if(EXISTS "${MUTLASS_LIBRARY_MANIFEST_CMAKE_FILE}")
  include(${MUTLASS_LIBRARY_MANIFEST_CMAKE_FILE})
else()
  message(STATUS "auto-generated library manifest cmake file (${MUTLASS_LIBRARY_MANIFEST_CMAKE_FILE}) not found.")
endif()

################################################################################

install(
  FILES ${MUTLASS_LIBRARY_GENERATED_KERNEL_LIST_FILE}
  DESTINATION ${CMAKE_INSTALL_INFODIR}/mutlass/
  )
