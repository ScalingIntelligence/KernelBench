#include <musa.h>
#include <musa_runtime.h>

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr unsigned char kCanaryValue = 0xA5;

struct GuardAllocatorOptions {
  bool shift_to_tail   = true;
  bool sync_on_free    = true;
  bool log_allocations = false;
};

struct AllocationRecord {
  uint64_t                     allocation_id    = 0;
  int                          device           = -1;
  size_t                       requested_size   = 0;
  size_t                       logical_size     = 0;
  size_t                       granularity      = 0;
  size_t                       mapped_size      = 0;
  size_t                       reservation_size = 0;
  bool                         shift_to_tail    = true;
  bool                         sync_on_free     = true;
  bool                         log_allocations  = false;
  MUdeviceptr                  reservation_ptr  = 0;
  MUdeviceptr                  mapped_ptr       = 0;
  MUdeviceptr                  logical_ptr      = 0;
  MUmemGenericAllocationHandle physical_handle  = 0;
};

std::mutex                                  g_mutex;
std::unordered_map<void*, AllocationRecord> g_allocations;
std::atomic<uint64_t>                       g_next_allocation_id{1};
GuardAllocatorOptions                       g_options;

[[noreturn]] void Fatal(const char* fmt, ...) {
  std::fprintf(stderr, "[mate.guard_allocator] ");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
  std::abort();
}

void Logf(const char* fmt, ...) {
  std::fprintf(stderr, "[mate.guard_allocator] ");
  va_list args;
  va_start(args, fmt);
  std::vfprintf(stderr, fmt, args);
  va_end(args);
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

void CheckRuntime(musaError_t err, const char* expr) {
  if (err != musaSuccess) {
    Fatal("runtime failure in %s: %s (%d)", expr, musaGetErrorString(err), static_cast<int>(err));
  }
}

void CheckDriver(MUresult err, const char* expr) {
  if (err != MUSA_SUCCESS) {
    const char* err_name = nullptr;
    const char* err_str  = nullptr;
    if (muGetErrorName(err, &err_name) != MUSA_SUCCESS || err_name == nullptr) {
      err_name = "<unknown>";
    }
    if (muGetErrorString(err, &err_str) != MUSA_SUCCESS || err_str == nullptr) {
      err_str = "<unknown>";
    }
    Fatal("driver failure in %s: %s (%s, %d)", expr, err_name, err_str, static_cast<int>(err));
  }
}

void CleanupReservation(MUdeviceptr reservation_ptr, size_t reservation_size) {
  if (reservation_ptr != 0 && reservation_size != 0) {
    (void)muMemAddressFree(reservation_ptr, reservation_size);
  }
}

void CleanupPhysicalHandle(MUmemGenericAllocationHandle handle) {
  if (handle != 0) {
    (void)muMemRelease(handle);
  }
}

void CleanupMappedAllocation(MUdeviceptr mapped_ptr, size_t mapped_size) {
  if (mapped_ptr != 0 && mapped_size != 0) {
    (void)muMemUnmap(mapped_ptr, mapped_size);
  }
}

void* DevicePtrToVoid(MUdeviceptr ptr) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
}

MUdeviceptr VoidToDevicePtr(void* ptr) {
  return static_cast<MUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
}

size_t AlignUp(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

GuardAllocatorOptions GetOptions() {
  const std::lock_guard<std::mutex> lock(g_mutex);
  return g_options;
}

void FillCanaryRegion(MUdeviceptr begin, size_t size) {
  if (size == 0) {
    return;
  }
  CheckRuntime(musaMemset(DevicePtrToVoid(begin), kCanaryValue, size), "musaMemset(canary)");
}

void VerifyCanary(const AllocationRecord& record) {
  size_t      canary_size  = 0;
  MUdeviceptr canary_begin = 0;
  if (record.shift_to_tail) {
    canary_begin = record.mapped_ptr;
    canary_size  = record.mapped_size - record.logical_size;
  } else {
    canary_begin = record.logical_ptr + record.logical_size;
    canary_size  = record.mapped_size - record.logical_size;
  }
  if (canary_size == 0) {
    return;
  }

  std::vector<unsigned char> host(canary_size);
  CheckRuntime(musaMemcpy(host.data(), DevicePtrToVoid(canary_begin), canary_size, musaMemcpyDeviceToHost),
               "musaMemcpy(canary -> host)");
  for (size_t i = 0; i < canary_size; ++i) {
    if (host[i] != kCanaryValue) {
      Fatal("allocation #%llu detected canary corruption at byte %zu (ptr=%p requested=%zu mode=%s)",
            static_cast<unsigned long long>(record.allocation_id),
            i,
            DevicePtrToVoid(record.logical_ptr),
            record.requested_size,
            record.shift_to_tail ? "tail" : "head");
    }
  }
}

AllocationRecord BuildAllocationRecord(size_t requested_size, int device, bool shift_to_tail) {
  CheckRuntime(musaSetDevice(device), "musaSetDevice");

  MUmemAllocationProp prop = {};
  prop.type                = MU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type       = MU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id         = device;
  prop.win32HandleMetaData = nullptr;

  size_t granularity = 0;
  CheckDriver(muMemGetAllocationGranularity(&granularity, &prop, MU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "muMemGetAllocationGranularity");

  AllocationRecord record;
  record.allocation_id    = g_next_allocation_id.fetch_add(1, std::memory_order_relaxed);
  record.device           = device;
  record.requested_size   = requested_size;
  record.logical_size     = requested_size == 0 ? 1 : requested_size;
  record.granularity      = granularity;
  record.mapped_size      = AlignUp(record.logical_size, granularity);
  record.reservation_size = record.mapped_size + granularity;
  record.shift_to_tail    = shift_to_tail;

  MUresult result = muMemAddressReserve(&record.reservation_ptr, record.reservation_size, 0, 0, 0);
  if (result == MUSA_ERROR_OUT_OF_MEMORY) {
    return record;
  }
  CheckDriver(result, "muMemAddressReserve");

  result = muMemCreate(&record.physical_handle, record.mapped_size, &prop, 0);
  if (result == MUSA_ERROR_OUT_OF_MEMORY) {
    CleanupReservation(record.reservation_ptr, record.reservation_size);
    record.reservation_ptr = 0;
    return record;
  }
  if (result != MUSA_SUCCESS) {
    CleanupReservation(record.reservation_ptr, record.reservation_size);
    record.reservation_ptr = 0;
    CheckDriver(result, "muMemCreate");
  }

  record.mapped_ptr = shift_to_tail ? record.reservation_ptr : (record.reservation_ptr + granularity);
  result            = muMemMap(record.mapped_ptr, record.mapped_size, 0, record.physical_handle, 0);
  if (result == MUSA_ERROR_OUT_OF_MEMORY) {
    CleanupPhysicalHandle(record.physical_handle);
    CleanupReservation(record.reservation_ptr, record.reservation_size);
    record.physical_handle = 0;
    record.reservation_ptr = 0;
    record.mapped_ptr      = 0;
    return record;
  }
  if (result != MUSA_SUCCESS) {
    CleanupPhysicalHandle(record.physical_handle);
    CleanupReservation(record.reservation_ptr, record.reservation_size);
    record.physical_handle = 0;
    record.reservation_ptr = 0;
    record.mapped_ptr      = 0;
    CheckDriver(result, "muMemMap");
  }

  MUmemAccessDesc access_desc = {};
  access_desc.location.type   = MU_MEM_LOCATION_TYPE_DEVICE;
  access_desc.location.id     = device;
  access_desc.flags           = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  result                      = muMemSetAccess(record.mapped_ptr, record.mapped_size, &access_desc, 1);
  if (result != MUSA_SUCCESS) {
    CleanupMappedAllocation(record.mapped_ptr, record.mapped_size);
    CleanupPhysicalHandle(record.physical_handle);
    CleanupReservation(record.reservation_ptr, record.reservation_size);
    record.physical_handle = 0;
    record.reservation_ptr = 0;
    record.mapped_ptr      = 0;
    CheckDriver(result, "muMemSetAccess");
  }

  size_t shift       = shift_to_tail ? (record.mapped_size - record.logical_size) : 0;
  record.logical_ptr = record.mapped_ptr + shift;
  return record;
}

bool IsAllocationOom(const AllocationRecord& record) {
  return record.reservation_ptr == 0 || record.physical_handle == 0 || record.mapped_ptr == 0;
}

void MaybeLogAllocation(const AllocationRecord& record, const char* action) {
  if (!record.log_allocations) {
    return;
  }
  Logf("%s allocation #%llu ptr=%p requested=%zu logical=%zu mapped=%zu granularity=%zu mode=%s device=%d",
       action,
       static_cast<unsigned long long>(record.allocation_id),
       DevicePtrToVoid(record.logical_ptr),
       record.requested_size,
       record.logical_size,
       record.mapped_size,
       record.granularity,
       record.shift_to_tail ? "tail" : "head",
       record.device);
}

}  // namespace

extern "C" void mate_guard_configure(int shift_to_tail, int sync_on_free, int log_allocations) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  g_options.shift_to_tail   = shift_to_tail != 0;
  g_options.sync_on_free    = sync_on_free != 0;
  g_options.log_allocations = log_allocations != 0;
}

extern "C" void* mate_guard_alloc(size_t size, int device, musaStream_t /*stream*/) {
  const GuardAllocatorOptions options = GetOptions();
  AllocationRecord            record  = BuildAllocationRecord(size, device, options.shift_to_tail);
  if (IsAllocationOom(record)) {
    return nullptr;
  }
  record.sync_on_free    = options.sync_on_free;
  record.log_allocations = options.log_allocations;

  if (record.shift_to_tail) {
    FillCanaryRegion(record.mapped_ptr, record.mapped_size - record.logical_size);
  } else {
    FillCanaryRegion(record.logical_ptr + record.logical_size, record.mapped_size - record.logical_size);
  }

  void* logical_ptr = DevicePtrToVoid(record.logical_ptr);
  {
    const std::lock_guard<std::mutex> lock(g_mutex);
    g_allocations.emplace(logical_ptr, record);
  }
  MaybeLogAllocation(record, "alloc");
  return logical_ptr;
}

extern "C" void mate_guard_free(void* ptr, size_t /*size*/, int /*device*/, musaStream_t /*stream*/) {
  AllocationRecord record;
  {
    const std::lock_guard<std::mutex> lock(g_mutex);
    auto                              it = g_allocations.find(ptr);
    if (it == g_allocations.end()) {
      Fatal("free received unknown pointer %p", ptr);
    }
    record = it->second;
    g_allocations.erase(it);
  }

  CheckRuntime(musaSetDevice(record.device), "musaSetDevice");
  if (record.sync_on_free) {
    CheckRuntime(musaDeviceSynchronize(), "musaDeviceSynchronize");
  }
  VerifyCanary(record);
  MaybeLogAllocation(record, "free");

  CheckDriver(muMemUnmap(record.mapped_ptr, record.mapped_size), "muMemUnmap");
  CheckDriver(muMemRelease(record.physical_handle), "muMemRelease");
  CheckDriver(muMemAddressFree(record.reservation_ptr, record.reservation_size), "muMemAddressFree");
}

extern "C" void* mate_guard_base_alloc(void* ptr, size_t* size) {
  const std::lock_guard<std::mutex> lock(g_mutex);
  auto                              it = g_allocations.find(ptr);
  if (it == g_allocations.end()) {
    if (size != nullptr) {
      *size = 0;
    }
    return ptr;
  }
  if (size != nullptr) {
    *size = it->second.mapped_size;
  }
  return DevicePtrToVoid(it->second.mapped_ptr);
}
