
#include "attention_scheduler.hpp"
#include "mate/attention/flash_mla/mpxx_get_mla_metadata.hpp"

void run_get_mla_metadata_kernel(mate::flash_mla::GetDecodingMetadataParams& params, musaStream_t stream) {
  int smem_size = sizeof(int) * (params.batch_size * 5 + 1);
  mate::flash_mla::get_mla_metadata_kernel<<<1, 32, smem_size, stream>>>(params);
}
