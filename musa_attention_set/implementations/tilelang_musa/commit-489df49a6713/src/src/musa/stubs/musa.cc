/**
 * \file musa.cc
 * \brief Implementation of MUSA driver API stub library.
 *
 * This file implements lazy loading of libmusa.so and provides global wrapper
 * functions that serve as drop-in replacements for the MUSA driver API.
 *
 * Motivation
 * ----------
 * The primary purpose is to allow TileLang to be imported on systems without
 * a GPU (e.g., CI/compilation nodes). The library is loaded on first API call
 * using dlopen(). If loading fails, an exception is thrown at call time rather
 * than at import time.
 */

#include "musa.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "musa_stub is currently POSIX-only (requires <dlfcn.h> / dlopen). "         \
    "On Windows, build TileLang from source with -DTILELANG_USE_MUSA_STUBS=OFF " \
    "to link against the real MUSA libraries."
#endif

#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace tvm::tl::musa {

namespace {

// Library names to try loading (in order of preference)
constexpr const char *kLibMusaPaths[] = {
    "libmusa.so.1", // Versioned library (most common)
    "libmusa.so",   // Unversioned library
};

/**
 * \brief Try to load libmusa.so from various paths.
 * \return The dlopen handle, or nullptr if loading failed.
 */
void *try_load_libmusa() {
  void *handle = nullptr;
  for (const char *path : kLibMusaPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      break;
    }
  }
  return handle;
}

/**
 * \brief Get symbol from library handle, returning nullptr on failure.
 */
template <typename T> T get_symbol(void *handle, const char *name) {
  // Clear any existing error
  (void)dlerror();
  void *sym = dlsym(handle, name);
  // Check for error (symbol could legitimately be nullptr in some cases)
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

/**
 * \brief Create and initialize the MUSADriverAPI singleton.
 *
 * This function loads libmusa.so and resolves all function symbols.
 * Required symbols that are missing will cause an exception.
 * Optional symbols that are missing will be set to nullptr.
 *
 * \return The initialized MUSADriverAPI instance.
 * \throws std::runtime_error if a required symbol is missing.
 */
MUSADriverAPI create_driver_api() {
  MUSADriverAPI api{};
  void *handle = MUSADriverAPI::get_handle();

  if (handle == nullptr) {
    return api;
  }

// Lookup required symbols - throw if missing
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);                    \
  if (api.name##_ == nullptr) {                                                \
    const char *error = dlerror();                                             \
    throw std::runtime_error(                                                  \
        std::string("Failed to load required MUSA driver symbol: ") + #name +  \
        ". Error: " + (error ? error : "unknown"));                            \
  }
  TILELANG_LIBMUSA_API_REQUIRED(LOOKUP_REQUIRED)
#undef LOOKUP_REQUIRED

// Lookup optional symbols - set to nullptr if missing (no throw)
#define LOOKUP_OPTIONAL(name)                                                  \
  api.name##_ = get_symbol<decltype(&name)>(handle, #name);
  TILELANG_LIBMUSA_API_OPTIONAL(LOOKUP_OPTIONAL)
#undef LOOKUP_OPTIONAL

  return api;
}

} // namespace

void *MUSADriverAPI::get_handle() {
  // Static handle ensures library is loaded only once
  static void *handle = try_load_libmusa();
  return handle;
}

bool MUSADriverAPI::is_available() { return get_handle() != nullptr; }

MUSADriverAPI *MUSADriverAPI::get() {
  static MUSADriverAPI singleton = create_driver_api();

  if (!is_available()) {
    throw std::runtime_error(
        "MUSA driver library (libmusa.so) not found. "
        "Please ensure MUSA drivers are installed, or use CPU-only mode.");
  }

  return &singleton;
}

} // namespace tvm::tl::musa

// ============================================================================
// Global wrapper function implementations
// ============================================================================
// These are the implementations for the extern "C" functions declared in the
// header. They provide ABI-compatible replacements for libmusa.so functions.

using tvm::tl::musa::MUSADriverAPI;

extern "C" {

MUresult muGetErrorName(MUresult error, const char **pStr) {
  return MUSADriverAPI::get()->muGetErrorName_(error, pStr);
}

MUresult muGetErrorString(MUresult error, const char **pStr) {
  return MUSADriverAPI::get()->muGetErrorString_(error, pStr);
}

MUresult muCtxGetDevice(MUdevice *device) {
  return MUSADriverAPI::get()->muCtxGetDevice_(device);
}

MUresult muCtxGetLimit(size_t *pvalue, MUlimit limit) {
  return MUSADriverAPI::get()->muCtxGetLimit_(pvalue, limit);
}

MUresult muCtxSetLimit(MUlimit limit, size_t value) {
  return MUSADriverAPI::get()->muCtxSetLimit_(limit, value);
}

MUresult muCtxResetPersistingL2Cache(void) {
  return MUSADriverAPI::get()->muCtxResetPersistingL2Cache_();
}

MUresult muDeviceGetName(char *name, int len, MUdevice dev) {
  return MUSADriverAPI::get()->muDeviceGetName_(name, len, dev);
}

MUresult muDeviceGetAttribute(int *pi, MUdevice_attribute attrib,
                              MUdevice dev) {
  return MUSADriverAPI::get()->muDeviceGetAttribute_(pi, attrib, dev);
}

MUresult muModuleLoadData(MUmodule *module, const void *image) {
  return MUSADriverAPI::get()->muModuleLoadData_(module, image);
}

MUresult muModuleLoadDataEx(MUmodule *module, const void *image,
                            unsigned int numOptions, MUjit_option *options,
                            void **optionValues) {
  return MUSADriverAPI::get()->muModuleLoadDataEx_(module, image, numOptions,
                                                   options, optionValues);
}

MUresult muModuleUnload(MUmodule hmod) {
  return MUSADriverAPI::get()->muModuleUnload_(hmod);
}

MUresult muModuleGetFunction(MUfunction *hfunc, MUmodule hmod,
                             const char *name) {
  return MUSADriverAPI::get()->muModuleGetFunction_(hfunc, hmod, name);
}

MUresult muModuleGetGlobal_v2(MUdeviceptr *dptr, size_t *bytes, MUmodule hmod,
                              const char *name) {
  return MUSADriverAPI::get()->muModuleGetGlobal_(dptr, bytes, hmod, name);
}

MUresult muFuncGetAttribute(int *pi, MUfunction_attribute attrib,
                            MUfunction hfunc) {
  return MUSADriverAPI::get()->muFuncGetAttribute_(pi, attrib, hfunc);
}

MUresult muFuncSetAttribute(MUfunction hfunc, MUfunction_attribute attrib,
                            int value) {
  return MUSADriverAPI::get()->muFuncSetAttribute_(hfunc, attrib, value);
}

MUresult muTensorDescriptorEncode(
    MUtensorDescriptor *tensorDesc, MUtensorDescriptorDataType tensorDataType,
    muuint32_t tensorRank, void *globalAddress, const muuint64_t *globalDim,
    const muuint64_t *globalStrides, MUtensorDescriptorInterleave interleave,
    muuint64_t oobConstantFill) {
  return MUSADriverAPI::get()->muTensorDescriptorEncode_(
      tensorDesc, tensorDataType, tensorRank, globalAddress, globalDim,
      globalStrides, interleave, oobConstantFill);
}

MUresult muLaunchKernel(MUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        MUstream hStream, void **kernelParams, void **extra) {
  return MUSADriverAPI::get()->muLaunchKernel_(
      f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, hStream, kernelParams, extra);
}

MUresult muLaunchKernelEx(const MUlaunchConfig *config, MUfunction f,
                          void **kernelParams, void **extra) {
  return MUSADriverAPI::get()->muLaunchKernelEx_(config, f, kernelParams,
                                                 extra);
}

MUresult muMemsetD32_v2(MUdeviceptr dstDevice, unsigned int ui, size_t N) {
  return MUSADriverAPI::get()->muMemsetD32_(dstDevice, ui, N);
}

MUresult muStreamSetAttribute(MUstream hStream, MUstreamAttrID attr,
                              const MUstreamAttrValue *value) {
  return MUSADriverAPI::get()->muStreamSetAttribute_(hStream, attr, value);
}

} // extern "C"
