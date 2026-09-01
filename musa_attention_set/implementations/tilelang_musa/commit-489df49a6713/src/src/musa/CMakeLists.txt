# MUSA target helpers are pure C++ registrations used by common TileLang code.
# Keep them available even when the MUSA toolchain is disabled.
list(APPEND TILE_LANG_SRCS src/musa/target_utils.cc)

if(NOT USE_MUSA)
  return()
endif()

message(STATUS "MUSA Backend is enabled")

if(DEFINED ENV{MUSA_HOME})
  set(MUSA_HOME $ENV{MUSA_HOME})
else()
  set(MUSA_HOME "/usr/local/musa")
endif()
list(APPEND CMAKE_MODULE_PATH ${MUSA_HOME}/cmake)

if(TILELANG_USE_MUSA_STUBS)
  set(MUSA_musart_LIBRARY "${CMAKE_BINARY_DIR}/lib/libstub_musart.so" CACHE FILEPATH
      "MUSA runtime library to link against" FORCE)
  set(MUSA_musa_driver_LIBRARY "${CMAKE_BINARY_DIR}/lib/libstub_musa.so" CACHE FILEPATH
      "MUSA driver library to link against" FORCE)
endif()

find_package(MUSAToolkit REQUIRED)
if(NOT MUSAToolkit_FOUND)
  message(FATAL_ERROR "Cannot find MUSA, USE_MUSA=" ${USE_MUSA})
endif()
include_directories(${MUSAToolkit_INCLUDE_DIRS})
list(APPEND TILE_LANG_INCLUDES ${MUSAToolkit_INCLUDE_DIRS})

if(TILELANG_USE_MUSA_STUBS)
  if(WIN32 AND NOT CYGWIN)
    message(FATAL_ERROR "TILELANG_USE_MUSA_STUBS=ON is not supported on Windows. "
                        "Please configure with -DTILELANG_USE_MUSA_STUBS=OFF.")
  endif()

  add_library(musa_stub SHARED src/musa/stubs/musa.cc)
  target_include_directories(musa_stub PRIVATE ${MUSAToolkit_INCLUDE_DIRS})
  target_compile_definitions(musa_stub PRIVATE TILELANG_MUSA_STUB_EXPORTS)
  target_link_libraries(musa_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(musa_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "stub_musa"
  )

  add_library(musart_stub SHARED src/musa/stubs/musart.cc)
  target_include_directories(musart_stub PRIVATE ${MUSAToolkit_INCLUDE_DIRS})
  target_link_libraries(musart_stub PRIVATE ${CMAKE_DL_LIBS})
  set_target_properties(musart_stub PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    OUTPUT_NAME "stub_musart"
  )

  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_LINK musa_stub musart_stub)
  list(APPEND TILELANG_ACTIVE_BACKEND_STUB_TARGETS musa_stub musart_stub)
  set(TILELANG_ACTIVE_BACKEND_PATCHELF_REMOVE
      "libmusa.so.1;libmusa.so;libmusart.so.5;libmusart.so.4;libmusart.so")
endif()

list(APPEND TILE_LANG_SRCS
  src/musa/codegen/codegen_musa.cc
  src/musa/codegen/intrin_rule_musa.cc
  src/musa/layout/mp31_sqmma_layouts.cc
  src/musa/op/fast_divmod.cc
  src/musa/op/atomic_add.cc
  src/musa/op/atomic_reduce.cc
  src/musa/op/copy.cc
  src/musa/op/fill.cc
  src/musa/op/finalize_reducer.cc
  src/musa/op/gemm.cc
  src/musa/op/gemm/mp31_sqmma.cc
  src/musa/op/reduce.cc
  src/musa/op/scan.cc
  src/musa/op/transpose.cc
  src/musa/transform/lower_ldg_stg.cc
  src/musa/transform/lower_fast_divmod.cc
  src/musa/transform/lower_tme.cc
  src/musa/runtime.cc
)
