/**
 * \file musart.cc
 * \brief MUSA Runtime API stub library for lazy loading libmusart.so at
 * runtime.
 *
 * Motivation
 * ----------
 * The primary purpose is to resolve SONAME mismatches (e.g., libmusart.so.4
 * vs libmusart.so.5), allowing a single build to work across different MUSA
 * versions. This is achieved by reusing the MUSA runtime already loaded by
 * frameworks like PyTorch.
 *
 * This stub exports the subset of MUSA Runtime API entrypoints used by TVM in
 * this repository. The real libmusart is loaded lazily via dlopen() on first
 * API call, and symbols are resolved via dlsym().
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <musa_runtime_api.h>

#if defined(_WIN32) && !defined(__CYGWIN__)
#error "musart_stub is currently POSIX-only (requires <dlfcn.h> / dlopen). "       \
    "On Windows, build TileLang from source with -DTILELANG_USE_MUSA_STUBS=OFF " \
    "to link against the real MUSA libraries."
#endif

#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Export symbols with default visibility for the shared stub library.
#define TILELANG_MUSART_STUB_API __attribute__((visibility("default")))

namespace {

constexpr const char *kLibMusartPaths[] = {
    "libmusart.so",
    "libmusart.so.4",
    "libmusart.so.5",
};

using MusaGraphInstantiate = musaError_t (*)(
    musaGraphExec_t *pGraphExec, musaGraph_t graph, unsigned long long flags);

void *TryLoadLibMusart() {
  // First, check if the symbols are already available globally.
  // This handles cases where PyTorch or another library has already loaded
  // libmusart, making its symbols available in the global namespace.
  // We use a representative symbol like musaGetErrorString.
  // dlsym with RTLD_DEFAULT searches the global scope.
  void *sym = dlsym(RTLD_DEFAULT, "musaGetErrorString");
  if (sym != nullptr && sym != reinterpret_cast<void *>(&musaGetErrorString)) {
    return RTLD_DEFAULT;
  }
  sym = dlsym(RTLD_NEXT, "musaGetErrorString");
  if (sym != nullptr) {
    return RTLD_NEXT;
  }

  // Otherwise, attempt to dlopen the library directly.
  void *handle = nullptr;
  for (const char *path : kLibMusartPaths) {
    handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle != nullptr) {
      return handle;
    }
  }

  fprintf(stderr,
          "TileLang Error: libmusart symbols not found. "
          "Make sure PyTorch with MUSA is installed before using TileLang.\n");
  abort();
}

template <typename T> T GetSymbol(void *handle, const char *name) {
  (void)dlerror();
  void *sym = dlsym(handle, name);
  const char *error = dlerror();
  if (error != nullptr) {
    return nullptr;
  }
  return reinterpret_cast<T>(sym);
}

struct MUSARuntimeAPI {
  decltype(&::musaGetErrorName) musaGetErrorName_{nullptr};
  decltype(&::musaGetErrorString) musaGetErrorString_{nullptr};
  decltype(&::musaGetLastError) musaGetLastError_{nullptr};
  decltype(&::musaPeekAtLastError) musaPeekAtLastError_{nullptr};

  decltype(&::musaSetDevice) musaSetDevice_{nullptr};
  decltype(&::musaGetDevice) musaGetDevice_{nullptr};
  decltype(&::musaGetDeviceCount) musaGetDeviceCount_{nullptr};
  decltype(&::musaDeviceGetAttribute) musaDeviceGetAttribute_{nullptr};
  decltype(&::musaGetDeviceProperties) musaGetDeviceProperties_{nullptr};

  decltype(&::musaMemGetInfo) musaMemGetInfo_{nullptr};
  decltype(&::musaMalloc) musaMalloc_{nullptr};
  decltype(&::musaFree) musaFree_{nullptr};
  decltype(&::musaMallocHost) musaMallocHost_{nullptr};
  decltype(&::musaFreeHost) musaFreeHost_{nullptr};
  decltype(&::musaMemset) musaMemset_{nullptr};
  decltype(&::musaMemsetAsync) musaMemsetAsync_{nullptr};

  decltype(&::musaMemcpy) musaMemcpy_{nullptr};
  decltype(&::musaMemcpyAsync) musaMemcpyAsync_{nullptr};
  decltype(&::musaMemcpyPeerAsync) musaMemcpyPeerAsync_{nullptr};

  decltype(&::musaStreamCreate) musaStreamCreate_{nullptr};
  decltype(&::musaStreamCreateWithFlags) musaStreamCreateWithFlags_{nullptr};
  decltype(&::musaStreamDestroy) musaStreamDestroy_{nullptr};
  decltype(&::musaStreamSynchronize) musaStreamSynchronize_{nullptr};
  decltype(&::musaStreamWaitEvent) musaStreamWaitEvent_{nullptr};

  decltype(&::musaEventCreate) musaEventCreate_{nullptr};
  decltype(&::musaEventDestroy) musaEventDestroy_{nullptr};
  decltype(&::musaEventRecord) musaEventRecord_{nullptr};
  decltype(&::musaEventSynchronize) musaEventSynchronize_{nullptr};
  decltype(&::musaEventElapsedTime) musaEventElapsedTime_{nullptr};

  decltype(&::musaDeviceSynchronize) musaDeviceSynchronize_{nullptr};
  decltype(&::musaLaunchCooperativeKernel) musaLaunchCooperativeKernel_{
      nullptr};

  decltype(&::musaStreamBeginCapture) musaStreamBeginCapture_{nullptr};
  decltype(&::musaStreamEndCapture) musaStreamEndCapture_{nullptr};
  MusaGraphInstantiate musaGraphInstantiate_{nullptr};
  MusaGraphInstantiate musaGraphInstantiateWithFlags_{nullptr};
  decltype(&::musaGraphLaunch) musaGraphLaunch_{nullptr};
  decltype(&::musaGraphDestroy) musaGraphDestroy_{nullptr};
  decltype(&::musaGraphExecDestroy) musaGraphExecDestroy_{nullptr};

  decltype(&::musaIpcGetMemHandle) musaIpcGetMemHandle_{nullptr};
  decltype(&::musaIpcOpenMemHandle) musaIpcOpenMemHandle_{nullptr};
  decltype(&::musaIpcCloseMemHandle) musaIpcCloseMemHandle_{nullptr};

  // Not currently required by default build, but cheap to include for optional
  // contribs (e.g. vllm kernels).
  decltype(&::musaFuncSetAttribute) musaFuncSetAttribute_{nullptr};
};

void *GetLibMusartHandle() {
  static void *handle = TryLoadLibMusart();
  return handle;
}

musaError_t MissingLibraryError() { return musaErrorUnknown; }

const char *FallbackMusaErrorString(musaError_t error) {
  if (error == musaSuccess) {
    return "musaSuccess";
  }
  if (error == musaErrorUnknown) {
    return "musaErrorUnknown (MUSA runtime stub: libmusart not found)";
  }
  return "musaError (MUSA runtime stub: libmusart not found)";
}

MUSARuntimeAPI CreateMUSARuntimeAPI() {
  MUSARuntimeAPI api{};
  void *handle = GetLibMusartHandle();
#define LOOKUP_REQUIRED(name)                                                  \
  api.name##_ = GetSymbol<decltype(api.name##_)>(handle, #name);               \
  if (api.name##_ == nullptr) {                                                \
    return MUSARuntimeAPI{};                                                   \
  }

  // NOTE: musaGetErrorString is optional in the sense that we can provide a
  // fallback string, but when libmusart is present it should always exist.
  api.musaGetErrorString_ = GetSymbol<decltype(api.musaGetErrorString_)>(
      handle, "musaGetErrorString");

  LOOKUP_REQUIRED(musaGetErrorName)
  LOOKUP_REQUIRED(musaGetLastError)
  LOOKUP_REQUIRED(musaPeekAtLastError)
  LOOKUP_REQUIRED(musaSetDevice)
  LOOKUP_REQUIRED(musaGetDevice)
  LOOKUP_REQUIRED(musaGetDeviceCount)
  LOOKUP_REQUIRED(musaDeviceGetAttribute)
  LOOKUP_REQUIRED(musaGetDeviceProperties)
  LOOKUP_REQUIRED(musaMemGetInfo)
  LOOKUP_REQUIRED(musaMalloc)
  LOOKUP_REQUIRED(musaFree)
  LOOKUP_REQUIRED(musaMallocHost)
  LOOKUP_REQUIRED(musaFreeHost)
  LOOKUP_REQUIRED(musaMemset)
  LOOKUP_REQUIRED(musaMemsetAsync)
  LOOKUP_REQUIRED(musaMemcpy)
  LOOKUP_REQUIRED(musaMemcpyAsync)
  LOOKUP_REQUIRED(musaMemcpyPeerAsync)
  LOOKUP_REQUIRED(musaStreamCreate)
  LOOKUP_REQUIRED(musaStreamCreateWithFlags)
  LOOKUP_REQUIRED(musaStreamDestroy)
  LOOKUP_REQUIRED(musaStreamSynchronize)
  LOOKUP_REQUIRED(musaStreamWaitEvent)
  LOOKUP_REQUIRED(musaEventCreate)
  LOOKUP_REQUIRED(musaEventDestroy)
  LOOKUP_REQUIRED(musaEventRecord)
  LOOKUP_REQUIRED(musaEventSynchronize)
  LOOKUP_REQUIRED(musaEventElapsedTime)
  LOOKUP_REQUIRED(musaDeviceSynchronize)
  LOOKUP_REQUIRED(musaLaunchCooperativeKernel)
  LOOKUP_REQUIRED(musaStreamBeginCapture)
  LOOKUP_REQUIRED(musaStreamEndCapture)
  LOOKUP_REQUIRED(musaGraphInstantiate)
  api.musaGraphInstantiateWithFlags_ = GetSymbol<MusaGraphInstantiate>(
      handle, "musaGraphInstantiateWithFlags");
  LOOKUP_REQUIRED(musaGraphLaunch)
  LOOKUP_REQUIRED(musaGraphDestroy)
  LOOKUP_REQUIRED(musaGraphExecDestroy)
  LOOKUP_REQUIRED(musaIpcGetMemHandle)
  LOOKUP_REQUIRED(musaIpcOpenMemHandle)
  LOOKUP_REQUIRED(musaIpcCloseMemHandle)

  // Optional
  api.musaFuncSetAttribute_ = GetSymbol<decltype(api.musaFuncSetAttribute_)>(
      handle, "musaFuncSetAttribute");

#undef LOOKUP_REQUIRED

  return api;
}

MUSARuntimeAPI *GetMUSARuntimeAPI() {
  static MUSARuntimeAPI singleton = CreateMUSARuntimeAPI();
  return &singleton;
}

musaError_t GraphInstantiate(musaGraphExec_t *pGraphExec, musaGraph_t graph,
                             unsigned long long flags,
                             musaGraphNode_t *pErrorNode, char *pLogBuffer,
                             size_t bufferSize) {
  (void)pErrorNode;
  (void)pLogBuffer;
  (void)bufferSize;
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGraphInstantiateWithFlags_ != nullptr) {
    return api->musaGraphInstantiateWithFlags_(pGraphExec, graph, flags);
  }
  if (api->musaGraphInstantiate_ == nullptr) {
    if (pGraphExec != nullptr) {
      *pGraphExec = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaGraphInstantiate_(pGraphExec, graph, flags);
}

} // namespace

extern "C" {

TILELANG_MUSART_STUB_API const char *musaGetErrorName(musaError_t error) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGetErrorName_ != nullptr) {
    return api->musaGetErrorName_(error);
  }
  return "musaErrorUnknown";
}

TILELANG_MUSART_STUB_API const char *musaGetErrorString(musaError_t error) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGetErrorString_ != nullptr) {
    return api->musaGetErrorString_(error);
  }
  return FallbackMusaErrorString(error);
}

TILELANG_MUSART_STUB_API musaError_t musaGetLastError(void) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGetLastError_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaGetLastError_();
}

TILELANG_MUSART_STUB_API musaError_t musaPeekAtLastError(void) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaPeekAtLastError_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaPeekAtLastError_();
}

TILELANG_MUSART_STUB_API musaError_t musaSetDevice(int device) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaSetDevice_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaSetDevice_(device);
}

TILELANG_MUSART_STUB_API musaError_t musaGetDevice(int *device) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGetDevice_ == nullptr) {
    if (device != nullptr) {
      *device = 0;
    }
    return MissingLibraryError();
  }
  return api->musaGetDevice_(device);
}

TILELANG_MUSART_STUB_API musaError_t musaGetDeviceCount(int *count) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGetDeviceCount_ == nullptr) {
    if (count != nullptr) {
      *count = 0;
    }
    return MissingLibraryError();
  }
  return api->musaGetDeviceCount_(count);
}

TILELANG_MUSART_STUB_API musaError_t musaDeviceGetAttribute(int *value,
                                                            musaDeviceAttr attr,
                                                            int device) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaDeviceGetAttribute_ == nullptr) {
    if (value != nullptr) {
      *value = 0;
    }
    return MissingLibraryError();
  }
  return api->musaDeviceGetAttribute_(value, attr, device);
}

TILELANG_MUSART_STUB_API musaError_t
musaGetDeviceProperties(musaDeviceProp *prop, int device) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGetDeviceProperties_ == nullptr) {
    if (prop != nullptr) {
      memset(prop, 0, sizeof(*prop));
    }
    return MissingLibraryError();
  }
  return api->musaGetDeviceProperties_(prop, device);
}

TILELANG_MUSART_STUB_API musaError_t musaMemGetInfo(size_t *free,
                                                    size_t *total) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMemGetInfo_ == nullptr) {
    if (free != nullptr) {
      *free = 0;
    }
    if (total != nullptr) {
      *total = 0;
    }
    return MissingLibraryError();
  }
  return api->musaMemGetInfo_(free, total);
}

TILELANG_MUSART_STUB_API musaError_t musaMalloc(void **devPtr, size_t size) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMalloc_ == nullptr) {
    if (devPtr != nullptr) {
      *devPtr = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaMalloc_(devPtr, size);
}

TILELANG_MUSART_STUB_API musaError_t musaFree(void *devPtr) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaFree_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaFree_(devPtr);
}

TILELANG_MUSART_STUB_API musaError_t musaMallocHost(void **ptr, size_t size) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMallocHost_ == nullptr) {
    if (ptr != nullptr) {
      *ptr = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaMallocHost_(ptr, size);
}

TILELANG_MUSART_STUB_API musaError_t musaFreeHost(void *ptr) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaFreeHost_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaFreeHost_(ptr);
}

TILELANG_MUSART_STUB_API musaError_t musaMemset(void *devPtr, int value,
                                                size_t count) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMemset_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaMemset_(devPtr, value, count);
}

TILELANG_MUSART_STUB_API musaError_t musaMemsetAsync(void *devPtr, int value,
                                                     size_t count,
                                                     musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMemsetAsync_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaMemsetAsync_(devPtr, value, count, stream);
}

TILELANG_MUSART_STUB_API musaError_t musaMemcpy(void *dst, const void *src,
                                                size_t count,
                                                musaMemcpyKind kind) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMemcpy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaMemcpy_(dst, src, count, kind);
}

TILELANG_MUSART_STUB_API musaError_t musaMemcpyAsync(void *dst, const void *src,
                                                     size_t count,
                                                     musaMemcpyKind kind,
                                                     musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMemcpyAsync_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaMemcpyAsync_(dst, src, count, kind, stream);
}

TILELANG_MUSART_STUB_API musaError_t
musaMemcpyPeerAsync(void *dst, int dstDevice, const void *src, int srcDevice,
                    size_t count, musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaMemcpyPeerAsync_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaMemcpyPeerAsync_(dst, dstDevice, src, srcDevice, count,
                                   stream);
}

TILELANG_MUSART_STUB_API musaError_t musaStreamCreate(musaStream_t *pStream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamCreate_ == nullptr) {
    if (pStream != nullptr) {
      *pStream = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaStreamCreate_(pStream);
}

TILELANG_MUSART_STUB_API musaError_t
musaStreamCreateWithFlags(musaStream_t *pStream, unsigned int flags) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamCreateWithFlags_ == nullptr) {
    if (pStream != nullptr) {
      *pStream = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaStreamCreateWithFlags_(pStream, flags);
}

TILELANG_MUSART_STUB_API musaError_t musaStreamDestroy(musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaStreamDestroy_(stream);
}

TILELANG_MUSART_STUB_API musaError_t
musaStreamSynchronize(musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamSynchronize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaStreamSynchronize_(stream);
}

TILELANG_MUSART_STUB_API musaError_t musaStreamWaitEvent(musaStream_t stream,
                                                         musaEvent_t event,
                                                         unsigned int flags) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamWaitEvent_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaStreamWaitEvent_(stream, event, flags);
}

TILELANG_MUSART_STUB_API musaError_t musaEventCreate(musaEvent_t *event) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaEventCreate_ == nullptr) {
    if (event != nullptr) {
      *event = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaEventCreate_(event);
}

TILELANG_MUSART_STUB_API musaError_t musaEventDestroy(musaEvent_t event) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaEventDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaEventDestroy_(event);
}

TILELANG_MUSART_STUB_API musaError_t musaEventRecord(musaEvent_t event,
                                                     musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaEventRecord_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaEventRecord_(event, stream);
}

TILELANG_MUSART_STUB_API musaError_t musaEventSynchronize(musaEvent_t event) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaEventSynchronize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaEventSynchronize_(event);
}

TILELANG_MUSART_STUB_API musaError_t musaEventElapsedTime(float *ms,
                                                          musaEvent_t start,
                                                          musaEvent_t end) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaEventElapsedTime_ == nullptr) {
    if (ms != nullptr) {
      *ms = 0.0f;
    }
    return MissingLibraryError();
  }
  return api->musaEventElapsedTime_(ms, start, end);
}

TILELANG_MUSART_STUB_API musaError_t musaDeviceSynchronize(void) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaDeviceSynchronize_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaDeviceSynchronize_();
}

TILELANG_MUSART_STUB_API musaError_t musaLaunchCooperativeKernel(
    const void *func, dim3 gridDim, dim3 blockDim, void **args,
    size_t sharedMem, musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaLaunchCooperativeKernel_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaLaunchCooperativeKernel_(func, gridDim, blockDim, args,
                                           sharedMem, stream);
}

TILELANG_MUSART_STUB_API musaError_t
musaStreamBeginCapture(musaStream_t stream, musaStreamCaptureMode mode) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamBeginCapture_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaStreamBeginCapture_(stream, mode);
}

TILELANG_MUSART_STUB_API musaError_t musaStreamEndCapture(musaStream_t stream,
                                                          musaGraph_t *graph) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaStreamEndCapture_ == nullptr) {
    if (graph != nullptr) {
      *graph = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaStreamEndCapture_(stream, graph);
}

TILELANG_MUSART_STUB_API musaError_t musaGraphInstantiate(
    musaGraphExec_t *pGraphExec, musaGraph_t graph, unsigned long long flags) {
  return GraphInstantiate(pGraphExec, graph, flags, nullptr, nullptr, 0);
}

TILELANG_MUSART_STUB_API musaError_t musaGraphLaunch(musaGraphExec_t graphExec,
                                                     musaStream_t stream) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGraphLaunch_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaGraphLaunch_(graphExec, stream);
}

TILELANG_MUSART_STUB_API musaError_t musaGraphDestroy(musaGraph_t graph) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGraphDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaGraphDestroy_(graph);
}

TILELANG_MUSART_STUB_API musaError_t
musaGraphExecDestroy(musaGraphExec_t graphExec) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaGraphExecDestroy_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaGraphExecDestroy_(graphExec);
}

TILELANG_MUSART_STUB_API musaError_t
musaIpcGetMemHandle(musaIpcMemHandle_t *handle, void *devPtr) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaIpcGetMemHandle_ == nullptr) {
    if (handle != nullptr) {
      memset(handle, 0, sizeof(*handle));
    }
    return MissingLibraryError();
  }
  return api->musaIpcGetMemHandle_(handle, devPtr);
}

TILELANG_MUSART_STUB_API musaError_t musaIpcOpenMemHandle(
    void **devPtr, musaIpcMemHandle_t handle, unsigned int flags) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaIpcOpenMemHandle_ == nullptr) {
    if (devPtr != nullptr) {
      *devPtr = nullptr;
    }
    return MissingLibraryError();
  }
  return api->musaIpcOpenMemHandle_(devPtr, handle, flags);
}

TILELANG_MUSART_STUB_API musaError_t musaIpcCloseMemHandle(void *devPtr) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaIpcCloseMemHandle_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaIpcCloseMemHandle_(devPtr);
}

TILELANG_MUSART_STUB_API musaError_t
musaFuncSetAttribute(const void *func, musaFuncAttribute attr, int value) {
  auto *api = GetMUSARuntimeAPI();
  if (api->musaFuncSetAttribute_ == nullptr) {
    return MissingLibraryError();
  }
  return api->musaFuncSetAttribute_(func, attr, value);
}

} // extern "C"
