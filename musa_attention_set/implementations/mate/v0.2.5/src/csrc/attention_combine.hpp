#pragma once

#include <musa_runtime.h>

#include "mate/attention/flash_mla/mpxx_params.hpp"

template <typename T, bool VarlenQ>
void run_mla_combine_kernel(const mate::flash_mla::MlaCombineParams& params, musaStream_t stream);
