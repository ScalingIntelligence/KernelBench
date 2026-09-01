#pragma once

#include <musa_runtime.h>

#include "mate/attention/flash_mla/mpxx_params.hpp"

void run_get_mla_metadata_kernel(mate::flash_mla::GetDecodingMetadataParams& params, musaStream_t stream);
