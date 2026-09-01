#pragma once

// MP31 TME device shim.
//
// Keep all MP31-specific instruction selection here; common MUSA templates
// must not include this header.  TME loads use the public MTCC __musa C++
// wrapper below rather than calling compiler builtins directly.

#include <musa.h>

#include "tl_templates/musa/common/intrin.h"

namespace tl {

#ifndef TL_DEVICE
#define TL_DEVICE __forceinline__ __device__
#endif

TL_DEVICE void tme_barrier_record(int barrier_id) {
  __musa::async_barrier barrier(barrier_id);
}

TL_DEVICE void tme_barrier_init_arrival(int barrier_id, int arrive_count,
                                        int phase_id = 0) {
  __musa::async_barrier barrier(barrier_id);
  barrier.init_arrival(arrive_count, phase_id);
}

TL_DEVICE void tme_barrier_add_trans(int barrier_id, int trans_count) {
  __musa::async_barrier barrier(barrier_id);
  barrier.add_trans(trans_count);
}

TL_DEVICE void tme_barrier_arrive(int barrier_id) {
  __musa::async_barrier barrier(barrier_id);
  barrier.arrive();
}

TL_DEVICE void tme_barrier_wait(int barrier_id, int phase_id) {
  __musa::async_barrier barrier(barrier_id);
  barrier.wait(phase_id);
}

TL_DEVICE void prefetch_tma_descriptor(
    const MUtensorDescriptor &descriptor) {
  // The installed MTCC exposes descriptor prefetch through the public
  // prefetch wrapper; the newer tme_desc_prefetch wrapper is not available
  // in this toolkit yet.
  ::prefetch(&descriptor);
}

TL_DEVICE void tme_load_im2col(const MUtensorDescriptor &descriptor,
                               int32_t barrier_id, void *smem_ptr,
                               int32_t range_c, int32_t range_npq, int32_t c,
                               int32_t q, int32_t p, int32_t n,
                               int32_t weight_pos, int32_t output_p,
                               int32_t output_q, int32_t conv_padding,
                               int32_t conv_stride, int32_t conv_dilation) {
  __musa::async_barrier barrier(barrier_id);
  __musa::memcpy_async(barrier, smem_ptr, &descriptor,
                       __musa::i2{range_c, range_npq}, __musa::i4{c, q, p, n},
                       0, weight_pos, __musa::i2{output_q, output_p},
                       __musa::i3{conv_padding, conv_stride, conv_dilation});
}

TL_DEVICE void tme_load(const MUtensorDescriptor &descriptor,
                        uint32_t barrier_id, void *smem_ptr, int coord0,
                        int dim0) {
  __musa::async_barrier barrier(barrier_id);
  __musa::memcpy_async(barrier, smem_ptr, &descriptor, dim0, coord0, 0,
                       __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B,
                       __musa::SZ_NONE);
}

TL_DEVICE void tme_load(const MUtensorDescriptor &descriptor,
                        uint32_t barrier_id, void *smem_ptr, int coord0,
                        int coord1, int dim0, int dim1) {
  __musa::i2 coord = {coord0, coord1};
  __musa::i2 dims = {dim0, dim1};
  __musa::async_barrier barrier(barrier_id);
  __musa::memcpy_async(barrier, smem_ptr, &descriptor, dims, coord, 0,
                       __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B,
                       __musa::SZ_NONE);
}

TL_DEVICE void tme_load(const MUtensorDescriptor &descriptor,
                        uint32_t barrier_id, void *smem_ptr, int coord0,
                        int coord1, int coord2, int dim0, int dim1, int dim2) {
  __musa::i3 coord = {coord0, coord1, coord2};
  __musa::i3 dims = {dim0, dim1, dim2};
  __musa::async_barrier barrier(barrier_id);
  __musa::memcpy_async(barrier, smem_ptr, &descriptor, dims, coord, 0,
                       __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B,
                       __musa::SZ_NONE);
}

TL_DEVICE void tme_load(const MUtensorDescriptor &descriptor,
                        uint32_t barrier_id, void *smem_ptr, int coord0,
                        int coord1, int coord2, int coord3, int dim0,
                        int dim1, int dim2, int dim3) {
  __musa::i4 coord = {coord0, coord1, coord2, coord3};
  __musa::i4 dims = {dim0, dim1, dim2, dim3};
  __musa::async_barrier barrier(barrier_id);
  __musa::memcpy_async(barrier, smem_ptr, &descriptor, dims, coord, 0,
                       __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B,
                       __musa::SZ_NONE);
}

TL_DEVICE void tme_load(const MUtensorDescriptor &descriptor,
                        uint32_t barrier_id, void *smem_ptr, int coord0,
                        int coord1, int coord2, int coord3, int coord4,
                        int dim0, int dim1, int dim2, int dim3, int dim4) {
  __musa::i5 coord = {coord0, coord1, coord2, coord3, coord4};
  __musa::i5 dims = {dim0, dim1, dim2, dim3, dim4};
  __musa::async_barrier barrier(barrier_id);
  __musa::memcpy_async(barrier, smem_ptr, &descriptor, dims, coord, 0,
                       __musa::SG_NONE, __musa::SS_256B, __musa::SL_256B,
                       __musa::SZ_NONE);
}

TL_DEVICE void tme_store(const MUtensorDescriptor &descriptor,
                         const void *smem_ptr, int coord0, int dim0) {
  __musa::memcpy(smem_ptr, &descriptor, dim0, coord0, __musa::SG_NONE,
                 __musa::SS_256B, __musa::SL_256B);
}

TL_DEVICE void tme_store(const MUtensorDescriptor &descriptor,
                         const void *smem_ptr, int coord0, int coord1,
                         int dim0, int dim1) {
  __musa::i2 coord = {coord0, coord1};
  __musa::i2 dims = {dim0, dim1};
  __musa::memcpy(smem_ptr, &descriptor, dims, coord, __musa::SG_NONE,
                 __musa::SS_256B, __musa::SL_256B);
}

TL_DEVICE void tme_store(const MUtensorDescriptor &descriptor,
                         const void *smem_ptr, int coord0, int coord1,
                         int coord2, int dim0, int dim1, int dim2) {
  __musa::i3 coord = {coord0, coord1, coord2};
  __musa::i3 dims = {dim0, dim1, dim2};
  __musa::memcpy(smem_ptr, &descriptor, dims, coord, __musa::SG_NONE,
                 __musa::SS_256B, __musa::SL_256B);
}

TL_DEVICE void tme_store(const MUtensorDescriptor &descriptor,
                         const void *smem_ptr, int coord0, int coord1,
                         int coord2, int coord3, int dim0, int dim1, int dim2,
                         int dim3) {
  __musa::i4 coord = {coord0, coord1, coord2, coord3};
  __musa::i4 dims = {dim0, dim1, dim2, dim3};
  __musa::memcpy(smem_ptr, &descriptor, dims, coord, __musa::SG_NONE,
                 __musa::SS_256B, __musa::SL_256B);
}

TL_DEVICE void tme_store(const MUtensorDescriptor &descriptor,
                         const void *smem_ptr, int coord0, int coord1,
                         int coord2, int coord3, int coord4, int dim0,
                         int dim1, int dim2, int dim3, int dim4) {
  __musa::i5 coord = {coord0, coord1, coord2, coord3, coord4};
  __musa::i5 dims = {dim0, dim1, dim2, dim3, dim4};
  __musa::memcpy(smem_ptr, &descriptor, dims, coord, __musa::SG_NONE,
                 __musa::SS_256B, __musa::SL_256B);
}

TL_DEVICE void tme_store_commit() {
  // MTCC currently exposes no public __musa wrapper for the MP31 TME store
  // commit instruction, so keep this builtin at the architecture boundary.
  __musa_tme_store_commit();
}

TL_DEVICE void tme_store_read_wait() {
  // MTCC currently exposes no public __musa wrapper for the MP31 TME store
  // read-wait instruction, so keep this builtin at the architecture boundary.
  __musa_tme_store_read_wait();
}

} // namespace tl
