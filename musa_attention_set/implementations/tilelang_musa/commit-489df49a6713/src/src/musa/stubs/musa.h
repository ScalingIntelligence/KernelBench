/**
 * \file musa.h
 * \brief Stub library for lazy loading libmusa.so at runtime.
 *
 * Motivation
 * ----------
 * libmusa.so is the MUSA Driver API library. Linking directly against it
 * creates a strong dependency on the presence of the MUSA driver at build
 * time and runtime.
 *
 * This stub library allows TileLang to:
 * 1. Be imported on machines without the MUSA driver library present.
 * 2. Avoid versioning conflicts by loading the available libmusa.so
 * dynamically.
 *
 * This library provides drop-in replacements for MUSA driver API functions.
 * It allows TileLang to be imported without the MUSA driver library installed.
 * The actual libmusa.so is loaded lazily on first API call.
 *
 * Usage:
 *
 * 1. Link against libstub_musa.so instead of libmusa.so
 *
 * 2. Call MUSA driver API functions normally - they are provided as
 *    exported global functions with C linkage:
 *
 *    ```cpp
 *    #include "musa/stubs/musa.h"
 *    MUresult result = muModuleLoadData(&mod, image);
 *    ```
 *
 * 3. For advanced use, access the singleton directly:
 *
 *    ```cpp
 *    auto* api = tvm::tl::musa::MUSADriverAPI::get();
 *    bool available = tvm::tl::musa::MUSADriverAPI::is_available();
 *    ```
 */

#pragma once

#include_next <musa.h>

// Symbol visibility macros for shared library export
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef TILELANG_MUSA_STUB_EXPORTS
#define TILELANG_MUSA_STUB_API __declspec(dllexport)
#else
#define TILELANG_MUSA_STUB_API __declspec(dllimport)
#endif
#else
#define TILELANG_MUSA_STUB_API __attribute__((visibility("default")))
#endif

// X-macro for listing all required MUSA driver API functions.
// Format: _(function_name)
// These are the core functions used by TVM/tilelang MUSA runtime.
#define TILELANG_LIBMUSA_API_REQUIRED(_)                                       \
  _(muGetErrorName)                                                            \
  _(muGetErrorString)                                                          \
  _(muCtxGetDevice)                                                            \
  _(muCtxGetLimit)                                                             \
  _(muCtxSetLimit)                                                             \
  _(muCtxResetPersistingL2Cache)                                               \
  _(muDeviceGetName)                                                           \
  _(muDeviceGetAttribute)                                                      \
  _(muModuleLoadData)                                                          \
  _(muModuleLoadDataEx)                                                        \
  _(muModuleUnload)                                                            \
  _(muModuleGetFunction)                                                       \
  _(muModuleGetGlobal)                                                         \
  _(muFuncGetAttribute)                                                        \
  _(muFuncSetAttribute)                                                        \
  _(muTensorDescriptorEncode)                                                  \
  _(muLaunchKernel)                                                            \
  _(muLaunchKernelEx)                                                          \
  _(muMemsetD32)                                                               \
  _(muStreamSetAttribute)

// Optional APIs (may not exist in older drivers or specific configurations)
// These are loaded but may be nullptr if not available
#define TILELANG_LIBMUSA_API_OPTIONAL(_)

namespace tvm::tl::musa {

/**
 * \brief MUSA Driver API accessor struct with lazy loading support.
 *
 * This struct provides lazy loading of libmusa.so symbols at first use,
 * allowing TileLang to be imported without the MUSA driver library installed.
 * The library handle and function pointers are stored as static members
 * to ensure one-time initialization.
 *
 * Usage:
 *   MUresult result = MUSADriverAPI::get()->muModuleLoadData_(&module, image);
 *
 * Note: Function pointers have a trailing underscore to differentiate from
 * the macro-redefined names in musa.h (e.g., muModuleGetGlobal ->
 * muModuleGetGlobal_v2)
 */
struct TILELANG_MUSA_STUB_API MUSADriverAPI {
// Create function pointer members for each API function
// The trailing underscore avoids conflict with musa.h macros
#define CREATE_MEMBER(name) decltype(&name) name##_;
  TILELANG_LIBMUSA_API_REQUIRED(CREATE_MEMBER)
  TILELANG_LIBMUSA_API_OPTIONAL(CREATE_MEMBER)
#undef CREATE_MEMBER

  /**
   * \brief Get the singleton instance of MUSADriverAPI.
   *
   * On first call, this loads libmusa.so and resolves all symbols.
   * Subsequent calls return the cached instance.
   *
   * \return Pointer to the singleton MUSADriverAPI instance.
   * \throws std::runtime_error if libmusa.so cannot be loaded or
   *         required symbols are missing.
   */
  static MUSADriverAPI *get();

  /**
   * \brief Check if MUSA driver is available without throwing.
   *
   * \return true if libmusa.so can be loaded, false otherwise.
   */
  static bool is_available();

  /**
   * \brief Get the raw library handle for libmusa.so.
   *
   * \return The dlopen handle, or nullptr if not loaded.
   */
  static void *get_handle();
};

} // namespace tvm::tl::musa

// ============================================================================
// Global wrapper functions for lazy-loaded MUSA driver API
// ============================================================================
// These functions provide drop-in replacements for MUSA driver API calls.
// They are exported from the stub library and can be linked against directly.
// The implementations are in musa.cc.

extern "C" {

TILELANG_MUSA_STUB_API MUresult muGetErrorName(MUresult error,
                                               const char **pStr);
TILELANG_MUSA_STUB_API MUresult muGetErrorString(MUresult error,
                                                 const char **pStr);
TILELANG_MUSA_STUB_API MUresult muCtxGetDevice(MUdevice *device);
TILELANG_MUSA_STUB_API MUresult muCtxGetLimit(size_t *pvalue, MUlimit limit);
TILELANG_MUSA_STUB_API MUresult muCtxSetLimit(MUlimit limit, size_t value);
TILELANG_MUSA_STUB_API MUresult muCtxResetPersistingL2Cache(void);
TILELANG_MUSA_STUB_API MUresult muDeviceGetName(char *name, int len,
                                                MUdevice dev);
TILELANG_MUSA_STUB_API MUresult muDeviceGetAttribute(int *pi,
                                                     MUdevice_attribute attrib,
                                                     MUdevice dev);
TILELANG_MUSA_STUB_API MUresult muModuleLoadData(MUmodule *module,
                                                 const void *image);
TILELANG_MUSA_STUB_API MUresult muModuleLoadDataEx(MUmodule *module,
                                                   const void *image,
                                                   unsigned int numOptions,
                                                   MUjit_option *options,
                                                   void **optionValues);
TILELANG_MUSA_STUB_API MUresult muModuleUnload(MUmodule hmod);
TILELANG_MUSA_STUB_API MUresult muModuleGetFunction(MUfunction *hfunc,
                                                    MUmodule hmod,
                                                    const char *name);
TILELANG_MUSA_STUB_API MUresult muModuleGetGlobal_v2(MUdeviceptr *dptr,
                                                     size_t *bytes,
                                                     MUmodule hmod,
                                                     const char *name);
TILELANG_MUSA_STUB_API MUresult muFuncGetAttribute(int *pi,
                                                   MUfunction_attribute attrib,
                                                   MUfunction hfunc);
TILELANG_MUSA_STUB_API MUresult muFuncSetAttribute(MUfunction hfunc,
                                                   MUfunction_attribute attrib,
                                                   int value);
TILELANG_MUSA_STUB_API MUresult muTensorDescriptorEncode(
    MUtensorDescriptor *tensorDesc, MUtensorDescriptorDataType tensorDataType,
    muuint32_t tensorRank, void *globalAddress, const muuint64_t *globalDim,
    const muuint64_t *globalStrides, MUtensorDescriptorInterleave interleave,
    muuint64_t oobConstantFill);
TILELANG_MUSA_STUB_API MUresult muLaunchKernel(
    MUfunction f, unsigned int gridDimX, unsigned int gridDimY,
    unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY,
    unsigned int blockDimZ, unsigned int sharedMemBytes, MUstream hStream,
    void **kernelParams, void **extra);
TILELANG_MUSA_STUB_API MUresult muLaunchKernelEx(const MUlaunchConfig *config,
                                                 MUfunction f,
                                                 void **kernelParams,
                                                 void **extra);
TILELANG_MUSA_STUB_API MUresult muMemsetD32_v2(MUdeviceptr dstDevice,
                                               unsigned int ui, size_t N);
TILELANG_MUSA_STUB_API MUresult muStreamSetAttribute(
    MUstream hStream, MUstreamAttrID attr, const MUstreamAttrValue *value);

} // extern "C"
