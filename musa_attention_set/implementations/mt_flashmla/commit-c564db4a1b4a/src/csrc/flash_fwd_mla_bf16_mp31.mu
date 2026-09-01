#include "flash_fwd_mla_kernel_musa.h"

template void run_mha_fwd_splitkv_mla<mutlass::bfloat16_t, 576>(Flash_fwd_mla_params &params, musaStream_t stream);
