
#pragma once

#include <mute/arch/mma.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/config.hpp>

#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ == 310))
#define MUTE_ARCH_SQMMA_MP31_ENABLED
#endif

namespace mate {

// Wait previous in-flight SQMMA to complete
template <int N = 0>
MUTE_HOST_DEVICE void warpsquad_wait() {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED)
#if (__MUSACC_VER_MAJOR__ >= 5) || (__MUSACC_VER_MAJOR__ >= 4 && __MUSACC_VER_MINOR__ >= 4) || \
    (__MUSACC_VER_MAJOR__ >= 4 && __MUSACC_VER_MINOR__ >= 3 && __MUSACC_VER_PATCHLEVEL__ >= 6)
  static_assert(N <= 2);
  __musa_tce_wait_group(N);
#else
  __musa_sqmma_wait();
#endif
#else
  MUTE_INVALID_CONTROL_PATH("Attempting to use sqmma wait without MUTE_ARCH_SQMMA_MP31_ENABLED");
#endif
}

MUTE_HOST_DEVICE
void warpsquad_commit_batch() {
#if defined(MUTE_ARCH_SQMMA_MP31_ENABLED) && \
    ((__MUSACC_VER_MAJOR__ >= 5) ||          \
     ((__MUSACC_VER_MAJOR__ >= 4) && (__MUSACC_VER_MINOR__ >= 3) && (__MUSACC_VER_PATCHLEVEL__ >= 6)))
  __musa_tce_commit_group();
#endif
}

}  // namespace mate
