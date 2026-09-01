/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/* \file
   \brief Execution environment
*/

#include <iostream>
#include <stdexcept>

// Profiler includes
#include "mutlass/profiler/mutlass_profiler.h"
#include "mutlass/profiler/gemm_operation_profiler.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace mutlass {
namespace profiler {

/////////////////////////////////////////////////////////////////////////////////////////////////

MutlassProfiler::MutlassProfiler(
  Options const &options
): 
  options_(options) {

  operation_profilers_.emplace_back(new GemmOperationProfiler(options));

}

MutlassProfiler::~MutlassProfiler() {

}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Execute the program
int MutlassProfiler::operator()() {

  if (options_.cmdline.num_naked_args() > 0) {
    std::cerr << "Unknown args: \n";
    options_.cmdline.print_naked_args(std::cerr);
    std::cerr << "\n\n\n";

    print_usage_(std::cout);
    return 1;
  }

  if (options_.about.help) {
    if (options_.operation_kind == library::OperationKind::kInvalid) {
      print_usage_(std::cout);
    }
    else {
      for (auto & profiler : operation_profilers_) {
        if (profiler->kind() == options_.operation_kind) {
          profiler->print_usage(std::cout);
          profiler->print_examples(std::cout);
          return 0;
        }
      }
    }
    return 0;
  }
  else if (options_.about.version) {
    options_.about.print_version(std::cout);

    std::cout << std::endl;
    return 0;
  }
  else if (options_.about.device_info) {
    options_.device.print_device_info(std::cout);
    return 0;
  }

  if (options_.execution_mode == ExecutionMode::kProfile ||
    options_.execution_mode == ExecutionMode::kDryRun ||
    options_.execution_mode == ExecutionMode::kTrace) {

    // Profiles all operations
    return profile_();
  }
  else if (options_.execution_mode == ExecutionMode::kEnumerate) {
    // Enumerates all operations
    enumerate_();
  }
  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Enumerates all operations
void MutlassProfiler::enumerate_() {

}

/// Profiles all operations
int MutlassProfiler::profile_() {

  int result = 0;
  DeviceContext device_context;

  // For all profilers
  for (auto & profiler : operation_profilers_) {

    if (options_.operation_kind == library::OperationKind::kInvalid ||
      options_.operation_kind == profiler->kind()) {

      result = profiler->profile_all(options_, library::Singleton::get().manifest, device_context);

      if (result) {
        return result;
      }
    }
  }

  return result;
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Prints all options
void MutlassProfiler::print_usage_(std::ostream &out) {
  options_.print_usage(out);

  out << "\nOperations:\n\n";

  // For all profilers
  for (auto & profiler : operation_profilers_) {


    std::string kind_str = library::to_string(profiler->kind());

    size_t kAlignment = 40;
    size_t columns = 0;

    if (kind_str.size() < kAlignment) {
      columns = kAlignment - kind_str.size();
    }

    out << "     " << kind_str << std::string(columns, ' ') << profiler->description() << "\n";

  }

  out << "\n\nFor details about a particular function, specify the function name with --help.\n\nExample:\n\n"
    << "  $ mutlass_profiler --operation=Gemm --help\n\n"
  ;
}

/// Prints usage
void MutlassProfiler::print_options_(std::ostream &out) {
  options_.print_options(out);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace profiler
} // namespace mutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
