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

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if(mutlass_LOADED)
  # If MUTLASS has been previously fetched and loaded, don't do it again.
  return()
else()
  set(mutlass_LOADED ON)
  set(MUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "MUTLASS Repository Directory")
endif()

message(STATUS "CMake Version: ${CMAKE_VERSION}")

# To reduce duplicate version locations, parse the version out of the
# main versions.h file and reuse it here.
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/include/mutlass/version.h VERSION_FILE_CONTENTS)
string(REGEX MATCH "#define MUTLASS_MAJOR ([0-9]+)" _MUTLASS_VERSION_MAJOR "${VERSION_FILE_CONTENTS}")
set(_MUTLASS_VERSION_MAJOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define MUTLASS_MINOR ([0-9]+)" _MUTLASS_VERSION_MINOR "${VERSION_FILE_CONTENTS}")
set(_MUTLASS_VERSION_MINOR ${CMAKE_MATCH_1})
string(REGEX MATCH "#define MUTLASS_PATCH ([0-9]+)" _MUTLASS_VERSION_PATCH "${VERSION_FILE_CONTENTS}")
set(_MUTLASS_VERSION_PATCH ${CMAKE_MATCH_1})

message(STATUS "MUTLASS ${_MUTLASS_VERSION_MAJOR}.${_MUTLASS_VERSION_MINOR}.${_MUTLASS_VERSION_PATCH}")

## MUTLASS PROJECT #############################################################

project(MUTLASS VERSION ${_MUTLASS_VERSION_MAJOR}.${_MUTLASS_VERSION_MINOR}.${_MUTLASS_VERSION_PATCH} LANGUAGES CXX)

################################################################################

#
# MUTLASS requires C++17
#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
list(APPEND MUTLASS_MUSA_MCC_FLAGS -std=c++17 -fno-strict-aliasing -Od3)

set(MUTLASS_TEST_LEVEL "0" CACHE STRING "Level of tests to compile.")

find_package(Python3 3.5 COMPONENTS Interpreter REQUIRED)

################################################################################
set(MUTLASS_ENABLE_HEADERS_ONLY OFF CACHE BOOL "Enable only the header library")

if(MUTLASS_ENABLE_HEADERS_ONLY)
  set(MUTLASS_ENABLE_EXAMPLES_INIT OFF)
  set(MUTLASS_ENABLE_TOOLS_INIT ON)
  set(MUTLASS_ENABLE_LIBRARY_INIT OFF)
  set(MUTLASS_ENABLE_TESTS_INIT OFF)
  set(MUTLASS_ENABLE_EXPERIMENTAL_INIT OFF)
else()
  set(MUTLASS_ENABLE_EXAMPLES_INIT ON)
  set(MUTLASS_ENABLE_TOOLS_INIT ON)
  set(MUTLASS_ENABLE_LIBRARY_INIT ON)
  if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(MUTLASS_ENABLE_TESTS_INIT ON)
  else()
    set(MUTLASS_ENABLE_TESTS_INIT OFF)
  endif()
  set(MUTLASS_ENABLE_EXPERIMENTAL_INIT ON)
endif()


set(MUTLASS_ENABLE_EXAMPLES ${MUTLASS_ENABLE_EXAMPLES_INIT} CACHE BOOL "Enable MUTLASS Examples")
set(MUTLASS_ENABLE_TOOLS ${MUTLASS_ENABLE_TOOLS_INIT} CACHE BOOL "Enable MUTLASS Tools")
set(MUTLASS_ENABLE_LIBRARY ${MUTLASS_ENABLE_LIBRARY_INIT} CACHE BOOL "Enable MUTLASS Library")
set(MUTLASS_ENABLE_PROFILER ${MUTLASS_ENABLE_LIBRARY} CACHE BOOL "Enable MUTLASS Profiler")
set(MUTLASS_ENABLE_EXPERIMENTAL ${MUTLASS_ENABLE_EXPERIMENTAL_INIT} CACHE BOOL "Enable MUTLASS Experimental")

set(MUTLASS_ENABLE_TESTS ${MUTLASS_ENABLE_TESTS_INIT} CACHE BOOL "Enable MUTLASS Tests")
set(MUTLASS_ENABLE_GTEST_UNIT_TESTS ${MUTLASS_ENABLE_TESTS} CACHE BOOL "Enable MUTLASS GTest-based Unit Tests")
################################################################################

set(MUTLASS_MCC_ARCHS_SUPPORTED "")
list(APPEND MUTLASS_MCC_ARCHS_SUPPORTED 22 31)
set(MUTLASS_MCC_ARCHS ${MUTLASS_MCC_ARCHS_SUPPORTED} CACHE STRING "The MP architectures requested.")
set(MUTLASS_MCC_ARCHS_ENABLED ${MUTLASS_MCC_ARCHS} CACHE STRING "The MP architectures to build code for.")

################################################################################

if (MUTLASS_ENABLE_GTEST_UNIT_TESTS)
  find_package(GTest QUIET)
  if (NOT GTest_FOUND)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/googletest.cmake)
  endif()
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/MUSA.cmake)

###################################################################################################
#
# Configure CMake variables
#
###################################################################################################

message(STATUS "MUSA Compilation Architectures: ${MUTLASS_MCC_ARCHS_ENABLED}")

if (NOT (CMAKE_BUILD_TYPE OR CONFIGURATION_TYPES))
  # By default we want to build in Release mode to ensure that we're getting best performance.
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose build level" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo" "Release")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (DEFINED CMAKE_DEBUG_POSTFIX)
  set(MUTLASS_LIBRARY_DEBUG_POSTFIX_INIT ${CMAKE_DEBUG_POSTFIX})
else()
  set(MUTLASS_LIBRARY_DEBUG_POSTFIX_INIT .debug)
endif()
set(MUTLASS_LIBRARY_DEBUG_POSTFIX ${MUTLASS_LIBRARY_DEBUG_POSTFIX_INIT} CACHE STRING "Default postfix value for debug libraries")

list(APPEND MUTLASS_MUSA_MCC_FLAGS -DMUTLASS_VERSIONS_GENERATED)

if (${MUTLASS_MCC_VERBOSE})
  list(APPEND MUTLASS_MUSA_MCC_FLAGS -v)
endif()

#
# MUTLASS NAMESPACE
#
set(MUTLASS_NAMESPACE "mutlass" CACHE STRING "Top level namespace of MUTLASS")


################################################################################

function(mutlass_apply_musa_gencode_flags ARCH_FLAGS)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs MP_ARCHS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (__MP_ARCHS)
    set(ARCHS_ENABLED ${__MP_ARCHS})
  else()
    set(ARCHS_ENABLED ${MUTLASS_MCC_ARCHS_ENABLED})
  endif()

  set(MCC_ARCH_FLAGS)
  foreach(ARCH ${ARCHS_ENABLED})
    list(APPEND MCC_ARCH_FLAGS --offload-arch=mp_${ARCH})
  endforeach()

  set(${ARCH_FLAGS} ${MCC_ARCH_FLAGS} PARENT_SCOPE)
endfunction()

###################################################################################################
#
# Define build targets
#
###################################################################################################

add_library(MUTLASS INTERFACE)
set_target_properties(MUTLASS PROPERTIES EXPORT_NAME mutlass)

set(MUTLASS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "MUTLASS Header Library")

set(MUTLASS_GENERATOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/library CACHE INTERNAL "Location of generator scripts")

# The following utility directory is needed even if the tools build is disabled, so it exists here.
set(MUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tools/util/include CACHE INTERNAL "")

include_directories(${MUTLASS_INCLUDE_DIR})

target_compile_features(MUTLASS INTERFACE cxx_std_17)

if (NOT MUTLASS_NAMESPACE STREQUAL "mutlass")
  target_compile_definitions(MUTLASS INTERFACE MUTLASS_NAMESPACE=${MUTLASS_NAMESPACE})
endif()

if (NOT DEFINED MUTLASS_REVISION)

  find_package(Git QUIET)

  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    RESULT_VARIABLE MUTLASS_REVISION_RESULT
    OUTPUT_VARIABLE MUTLASS_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (MUTLASS_REVISION_RESULT)
    message(STATUS "MUTLASS Revision: Unable to detect, Git returned code ${MUTLASS_REVISION_RESULT}.")
  else()
    message(STATUS "MUTLASS Revision: ${MUTLASS_REVISION}")
  endif()

endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version_extended.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/mutlass/version_extended.h
  @ONLY)

target_include_directories(
  MUTLASS
  INTERFACE
  ${MUTLASS_INCLUDE_DIR}
)

################################################################################

include(CTest)
enable_testing()

if (NOT TARGET test_all)
  add_custom_target(test_all)
endif()


################################################################################
set(MUTLASS_CTEST_TEMPLATE_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestTestfile.configure.cmake)

function(mutlass_add_executable_tests NAME TARGET)
  set(options DISABLE_EXECUTABLE_INSTALL_RULE)
  set(oneValueArgs DISABLE_TESTS RESULT_CACHE_FILE TEST_COMMAND_OPTIONS_PREFIX)
  set(multiValueArgs DEPENDS DEPENDEES TEST_COMMAND_OPTIONS)
  cmake_parse_arguments(_ "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if (NOT DEFINED __DISABLE_TESTS)
    set(__DISABLE_TESTS OFF)
  endif()

  set(TEST_EXE $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_EXE_WORKING_DIRECTORY ./${CMAKE_INSTALL_BINDIR})

  if (__RESULT_CACHE_FILE)
    add_custom_command(
      TARGET ${TARGET}
      POST_BUILD
      COMMAND ${CMAKE_COMMAND}
      ARGS -E copy ${__RESULT_CACHE_FILE} "$<TARGET_FILE_DIR:${TARGET}>"
    )
  endif()

  if (NOT __DISABLE_EXECUTABLE_INSTALL_RULE AND MUTLASS_INSTALL_TESTS)
    install(
      TARGETS ${TARGET}
      RUNTIME DESTINATION ${MUTLASS_TEST_INSTALL_BINDIR}
    )
    if (__RESULT_CACHE_FILE)
      install(
        FILES ${__RESULT_CACHE_FILE}
        DESTINATION ${MUTLASS_TEST_INSTALL_BINDIR}/
      )
    endif()
  endif()

  if (NOT __TEST_COMMAND_OPTIONS)
    set(__TEST_COMMAND_OPTIONS " ")
  endif()

  list(LENGTH __TEST_COMMAND_OPTIONS CMD_COUNT)

  if (CMD_COUNT GREATER 1)
    add_custom_target(${NAME} DEPENDS ${TARGET} ${__DEPENDS})
    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${NAME})
    endforeach()
  endif()

  if (MUTLASS_INSTALL_TESTS)
    set(_INLINE_PER_TEST_CODE)
    file(READ "${PROJECT_SOURCE_DIR}/cmake/CTestTestfile.test.configure.cmake" _INLINE_PER_TEST_CODE_TEMPLATE)
  endif()

  set(TEST_GROUP_NAME ${NAME})

  foreach(CMD_OPTIONS_VAR IN LISTS __TEST_COMMAND_OPTIONS)
    if (CMD_COUNT GREATER 1)
      string(TOLOWER "${NAME}_${CMD_OPTIONS_VAR}" TEST_NAME)
    else()
      string(TOLOWER "${NAME}" TEST_NAME)
    endif()

    set(TEST_COMMAND_OPTIONS ${${__TEST_COMMAND_OPTIONS_PREFIX}${CMD_OPTIONS_VAR}})
    list(JOIN TEST_COMMAND_OPTIONS " " TEST_COMMAND_OPTIONS)
    separate_arguments(TEST_COMMAND_OPTIONS)

    add_custom_target(
      ${TEST_NAME}
      COMMAND
      ${MUTLASS_TEST_EXECUTION_ENVIRONMENT} $<TARGET_FILE:${TARGET}> ${TEST_COMMAND_OPTIONS}
      DEPENDS
      ${TARGET}
    )

    if (CMD_COUNT GREATER 1)
      add_dependencies(${NAME} ${TEST_NAME})
    endif()

    foreach(DEPENDEE ${__DEPENDEES})
      add_dependencies(${DEPENDEE} ${TEST_NAME})
    endforeach()

    set(TEST_NAME c${TEST_NAME})

    string(CONFIGURE "${_INLINE_PER_TEST_CODE_TEMPLATE}" _TEST_CODE @ONLY)
    string(APPEND _INLINE_PER_TEST_CODE "${_TEST_CODE}")
  endforeach()

  set(TEST_NAME c${NAME})
  set(TEST_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/ctest/${TEST_NAME})
  file(MAKE_DIRECTORY ${TEST_GEN_DIR})

  set(TEST_SETS_SUPPORTED default)

  set(TEST_EXE_PATH $<TARGET_FILE:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT ON)
  configure_file("${MUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.cmake" @ONLY)

  set(TEST_EXE_PATH $<TARGET_FILE_NAME:${TARGET}>)
  set(TEST_USE_EXTENDED_FORMAT OFF)
  configure_file("${MUTLASS_CTEST_TEMPLATE_FILE}" "${TEST_GEN_DIR}/CTestTestfile.${TEST_NAME}.install.cmake.in" @ONLY)
endfunction()

if (MUTLASS_ENABLE_TOOLS)
  add_subdirectory(tools)
  if (MUTLASS_ENABLE_PROFILER)
    add_dependencies(test_all test_profiler)
  endif()
endif()

if (MUTLASS_ENABLE_EXAMPLES)
  add_subdirectory(examples)
  add_dependencies(test_all test_examples)
endif()

if (MUTLASS_ENABLE_TESTS)
  add_subdirectory(test)
  if (MUTLASS_ENABLE_GTEST_UNIT_TESTS)
    add_dependencies(test_all test_unit)
  endif()
endif()

if (MUTLASS_ENABLE_EXPERIMENTAL)
  add_subdirectory(experimental)
  add_dependencies(test_all test_experimental)
endif()
