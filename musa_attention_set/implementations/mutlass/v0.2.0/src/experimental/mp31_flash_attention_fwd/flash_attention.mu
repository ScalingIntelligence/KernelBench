/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Moore Threads Technology Co., Ltd("Moore Threads"). All rights reserved.
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

#include <mutlass/mutlass.h>

#include <mutlass/util/command_line.h>
#include <mutlass/util/GPU_Clock.hpp>
#include <mutlass/util/device_memory.h>
#include <mutlass/util/distribution.h>
#include <mutlass/util/reference/device/tensor_fill.h>
#include <mutlass/util/reference/host/tensor_fill.h>
#include <mutlass/util/GPU_Clock.hpp>

#include <helper.h>

#include <vector>

#include "kernel_traits.hpp"
#include "fwd_params.hpp"
#include "flash_attn_fwd.hpp"
#include "reference_attention.hpp"

///////////////////////////////////////////////////////////////////////////////////////////////////
struct Options {
  bool help;
  bool verbose;
  bool skip_check;
  int q_seq_len, kv_seq_len, batch_size, dim_sizeqk, dim_sizev, nheads, head_group;
  int warmup_iter, benchmark_iter; 
  mutlass::Distribution::Kind dist_kind;

  Options():
    help(false),
    verbose(false),
    skip_check(false),
    dist_kind(mutlass::Distribution::Kind::Uniform),
    warmup_iter(1), benchmark_iter(1),
    batch_size(1), dim_sizeqk(128),dim_sizev(128),
    q_seq_len(128), kv_seq_len(128), nheads(1), head_group(1)
  { }

  void parse(int argc, char const** args) {
    mutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }
    if (cmd.check_cmd_line_flag("verbose")) {
      verbose = true;
    }
    if (cmd.check_cmd_line_flag("skip-check")) {
      skip_check = true;
    }

    std::string dist_str;
    cmd.get_cmd_line_argument("dist", dist_str);
    if (dist_str == "gaussian") {
      dist_kind = mutlass::Distribution::Gaussian;
    }

    cmd.get_cmd_line_argument("batch-size", batch_size, 1);
    cmd.get_cmd_line_argument("dim-sizeqk", dim_sizeqk, 576);
    cmd.get_cmd_line_argument("dim-sizev", dim_sizev, 512);
    cmd.get_cmd_line_argument("q-seq-len", q_seq_len, 4096);
    cmd.get_cmd_line_argument("kv-seq-len", kv_seq_len, 4096);
    cmd.get_cmd_line_argument("nheads", nheads, 1);
    cmd.get_cmd_line_argument("head-group", head_group, 1);
    cmd.get_cmd_line_argument("warmup", warmup_iter, 1);
    cmd.get_cmd_line_argument("bench", benchmark_iter, 1);
  }

  std::ostream & print_usage(std::ostream &out) const {
    out << "FlashAttention Forward:\n\n"
        << "Options:\n"
        << "  --help                      If specified, displays this usage statement.\n"
        << "  --verbose                   If specified, displays detail information.\n"
        << "  --skip-check                If specified, skip the correctnes check\n"
        << "  --dist                      Data distribution of input tensors.(--dist=uniform/gaussian, default: uniform)\n"
        << "  --warmup=<int>              Warmup iterations(default: --warmup=1)\n"
        << "  --bench=<int>               Benchmark iterations(default: --bench=1)\n"
        << "  --batch-size=<int>          Batch size in multi-head attention "
           "(default: --batch_size=1).\n"
        << "  --dim-sizeqk=<int>            Full Size of the QK head dimension "
           "(before reshape, default: --dim-size=128). \n"
        << "  --dim-sizev=<int>            Full Size of the V head dimension "
           "(before reshape, default: --dim-size=128). \n"
        << "  --q-seq-len=<int>           Sequence length in multi-head "
           "attention for Q (default: --q-seq-len=256).\n"
        << "  --kv-seq-len=<int>          Sequence length in multi-head "
           "attention for KV (default: --kv-seq-len=256).\n"
        << "  --nheads=<int>              Number of heads in multi-head attention "
           "(default: --nheads=1).\n"
        << "  --head-group=<int>          Number of heads in multi-head attention "
           "(default: --head-group=1).\n"
        << std::endl;
    return out;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element>
bool initialize_block(
    mutlass::DeviceAllocation<Element>& block,
    mutlass::Distribution::Kind dist_kind = mutlass::Distribution::Kind::Uniform,
    uint64_t seed=2024) {

  if (dist_kind == mutlass::Distribution::Uniform) {
    Element scope_max, scope_min;
    int bits_input = mutlass::sizeof_bits<Element>::value;
    if (bits_input == 1) {
      scope_max = Element(2);
      scope_min = Element(0);
    } else if (bits_input <= 8) {
      scope_max = Element(2);
      scope_min = Element(-2);
    } else {
      scope_max = Element(2);
      scope_min = Element(-2);
    }
    mutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, scope_max, scope_min, 0);
  } else if (dist_kind == mutlass::Distribution::Gaussian) {
    mutlass::reference::device::BlockFillRandomGaussian(
      block.get(), block.size(), seed, Element(0.0f), Element(1.0f));
  }
  return true;
}



///////////////////////////////////////////////////////////////////////////////////////////////////

template <class KernelTraits>
struct FlashAttentionFwdRunner {
  using Element = typename KernelTraits::Element;
  using ElementAccum = typename KernelTraits::ElementAccum;
  using FwdParams = CollectiveFwdParams<KernelTraits>;

  static constexpr int HeadDimQK = KernelTraits::HeadDimQK;
  static constexpr int HeadDimV = KernelTraits::HeadDimV;

  // Data members
  mutlass::DeviceAllocation<Element> block_Q;
  mutlass::DeviceAllocation<Element> block_K;
  mutlass::DeviceAllocation<Element> block_V;
  mutlass::DeviceAllocation<Element> block_O;
  mutlass::DeviceAllocation<Element> block_S;

  using index_t = int64_t;

  index_t q_batch_stride;
  index_t k_batch_stride;
  index_t v_batch_stride;
  index_t o_batch_stride;
  index_t s_batch_stride;
  index_t q_row_stride;
  index_t k_row_stride;
  index_t v_row_stride;
  index_t o_row_stride;
  index_t s_row_stride;
  index_t q_head_stride;
  index_t k_head_stride;
  index_t v_head_stride;
  index_t o_head_stride;
  index_t s_head_stride;
  int hq;
  int hkv;

  void setup_stride(const Options& options) {
    // Q stride
    q_row_stride   = options.dim_sizeqk;
    q_head_stride  = options.dim_sizeqk * options.q_seq_len;
    q_batch_stride = options.dim_sizeqk * options.q_seq_len * hq;

    // K stride
    k_row_stride   = options.dim_sizeqk;
    k_head_stride  = options.dim_sizeqk * options.kv_seq_len;
    k_batch_stride = options.dim_sizeqk * options.kv_seq_len * hkv;

    // V stride
    v_row_stride   = options.dim_sizev;
    v_head_stride  = options.dim_sizev * options.kv_seq_len;
    v_batch_stride = options.dim_sizev * options.kv_seq_len * hkv;

    // O stride
    o_row_stride   = options.dim_sizev;
    o_head_stride  = options.dim_sizev * options.q_seq_len;
    o_batch_stride = options.dim_sizev * options.q_seq_len * hq;

    // S stride
    s_row_stride   = options.kv_seq_len;
    s_head_stride  = options.kv_seq_len * options.q_seq_len;
    s_batch_stride = s_head_stride * hq;
  }

  void initialize(const Options& options) {
    block_Q.reset(options.q_seq_len  * options.dim_sizeqk * hq * options.batch_size);
    block_K.reset(options.kv_seq_len * options.dim_sizeqk * hkv * options.batch_size);
    block_V.reset(options.kv_seq_len * options.dim_sizev * hkv * options.batch_size);
    block_O.reset(options.q_seq_len  * options.dim_sizev * hq * options.batch_size);
    block_S.reset(options.q_seq_len  * options.kv_seq_len * hq * options.batch_size);

    // initialize_block(block_Q, options.dist_kind);
    // initialize_block(block_K, options.dist_kind);
    // initialize_block(block_V, options.dist_kind);

    initialize_block(block_Q, mutlass::Distribution::Gaussian);
    initialize_block(block_K, mutlass::Distribution::Gaussian);
    initialize_block(block_V, mutlass::Distribution::Gaussian);
  }

  template <bool EnableSoftmax=true, bool WithCausal=true>
  bool verify(const Options& options) {
    TestAttention<Element, ElementAccum> testBed(hq, hkv, options.batch_size, options.dim_sizeqk, options.dim_sizev, 
                                                 options.q_seq_len, options.kv_seq_len, 1,false,WithCausal);
    testBed.initialize();

    std::vector<ElementAccum> MiOut(options.q_seq_len * hq * options.batch_size);
    std::vector<ElementAccum> SprimeOut(options.q_seq_len * hq * options.batch_size);

    std::vector<Element> gpu_O(block_O.capacity);
    std::vector<Element> ref_O = gpu_O;

    block_O.copy_to_host(gpu_O.data());

    constexpr bool use_pre_scaling = EnableSoftmax ? true : false;
    constexpr bool use_pow2 = true;

    #if DEVICE_REF
    printf("Device Reference...\n");
    mutlass::DeviceAllocation<Element> block_S_ref(block_S.capacity);
    mutlass::DeviceAllocation<Element> block_O_ref(block_O.capacity);

    testBed.compute(block_Q.get(), block_K.get(), block_V.get(),
                    block_S_ref.get(), block_O_ref.get(),
                    MiOut.data(), SprimeOut.data(),
                    use_pow2, use_pre_scaling, EnableSoftmax);

    block_O_ref.copy_to_host(ref_O.data());
    musaDeviceSynchronize();

    #else
    printf("Host Reference...\n");
    std::vector<Element> host_Q(block_Q.capacity);
    std::vector<Element> host_K(block_K.capacity);
    std::vector<Element> host_V(block_V.capacity);

    std::vector<Element> ref_S(options.q_seq_len * options.kv_seq_len * hq * options.batch_size);

    block_Q.copy_to_host(host_Q.data());
    block_K.copy_to_host(host_K.data());
    block_V.copy_to_host(host_V.data());

    musaDeviceSynchronize();

    testBed.compute(host_Q.data(), host_K.data(), host_V.data(),
                    ref_S.data(), ref_O.data(),
                    MiOut.data(), SprimeOut.data(),
                    use_pow2, use_pre_scaling, EnableSoftmax);
    #endif
    return verify_tensor<Element>(gpu_O, ref_O, false, true);
  }

  template <bool EnableSoftmax=true, bool WithCausal=true>
  bool run(const Options& options) {
    hq = options.nheads;
    hkv = options.nheads / options.head_group;
    assert(options.nheads % options.head_group == 0);
    setup_stride(options);
    initialize(options);

    constexpr float kLog2e = float(1.4426950408889634074);
    const float softmax_scale = (1.0f / sqrt(float(options.dim_sizeqk)));
    float rln2_scale = (float)1.0;
    if constexpr (EnableSoftmax) {
      rln2_scale = softmax_scale * kLog2e;
    }

    typename FwdParams::Arguments arguments {
      block_Q.get(),
      FwdParams::get_gmem_layout(options.q_seq_len, options.dim_sizeqk, hq, options.batch_size,
                                 q_row_stride, q_head_stride, q_batch_stride),
      block_K.get(),
      FwdParams::get_gmem_layout(options.kv_seq_len, options.dim_sizeqk, hkv, options.batch_size,
                                 k_row_stride, k_head_stride, k_batch_stride),
      block_V.get(),
      FwdParams::get_gmem_layout(options.kv_seq_len, options.dim_sizev, hkv, options.batch_size,
                                 v_row_stride, v_head_stride, v_batch_stride),
      rln2_scale, ceil_div(options.kv_seq_len, KernelTraits::BlockN),
      block_O.get(),
      FwdParams::get_gmem_layout(options.q_seq_len, options.dim_sizev, hq, options.batch_size,
                                 o_row_stride, o_head_stride, o_batch_stride),
      block_S.get(),
      FwdParams::get_gmem_layout(options.q_seq_len, options.kv_seq_len, hq, options.batch_size,
                                 s_row_stride, s_head_stride, s_batch_stride)
    };

    typename FwdParams::Params params = FwdParams::to_underlying_arguments(arguments);

    dim3 block_dim(KernelTraits::NumThreads);
    dim3 grid_dims(ceil_div(options.q_seq_len, KernelTraits::BlockM), hq, options.batch_size);

    musaError_t error = musaDeviceSynchronize();
    if (error != musaSuccess) {
      std::cerr << "Somthing failed. Error is " << musaGetErrorString(error) << std::endl;
      return false;
    }

    flash_atten_fwd<Element, decltype(params.layout_Q), decltype(params.tme_load_Q),
                    decltype(params.layout_K), decltype(params.key_desc),
                    decltype(params.layout_V), decltype(params.tme_load_V),
                    decltype(params.layout_O), decltype(params.layout_S),
                    EnableSoftmax, WithCausal, KernelTraits>
        <<<grid_dims, block_dim>>>(
            params.layout_Q, params.tme_load_Q,
            params.layout_K, params.key_desc, params.ptr_K,
            params.layout_V, params.tme_load_V,
            params.layout_O, params.ptr_O,
            params.layout_S, params.ptr_S,
            params.n_tiles, params.rln2_scale);

    error = musaGetLastError();
    if (error != musaSuccess) {
      std::cerr << "Launch FA failed. Error is " << musaGetErrorString(error) << std::endl;
      return false;
    }

    error = musaDeviceSynchronize();
    if (error != musaSuccess) {
      std::cerr << "Running FA failed. Error is " << musaGetErrorString(error) << std::endl;
      return false;
    }

    if (!options.skip_check) {
      bool passed = verify<EnableSoftmax, WithCausal>(options);

      if (passed) {
        std::cout << "FlasAttn Fwd Passed!\n";
      } else {
        std::cerr << "FlasAttn Fwd Failed!\n";
        return false;
      }
    }

    for (int i = 0; i < options.warmup_iter; ++i)
    {
      flash_atten_fwd<Element, decltype(params.layout_Q), decltype(params.tme_load_Q),
                      decltype(params.layout_K), decltype(params.key_desc),
                      decltype(params.layout_V), decltype(params.tme_load_V),
                      decltype(params.layout_O), decltype(params.layout_S),
                      EnableSoftmax, WithCausal, KernelTraits>
          <<<grid_dims, block_dim>>>(
              params.layout_Q, params.tme_load_Q,
              params.layout_K, params.key_desc, params.ptr_K,
              params.layout_V, params.tme_load_V,
              params.layout_O, params.ptr_O,
              params.layout_S, params.ptr_S,
              params.n_tiles, params.rln2_scale);

      error = musaGetLastError();
      if (error != musaSuccess) {
        std::cerr << "Launch FA failed. Error is " << musaGetErrorString(error) << std::endl;
        return false;
      }
    }

    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < options.benchmark_iter; ++i)
    {
      flash_atten_fwd<Element, decltype(params.layout_Q), decltype(params.tme_load_Q),
                      decltype(params.layout_K), decltype(params.key_desc),
                      decltype(params.layout_V), decltype(params.tme_load_V),
                      decltype(params.layout_O), decltype(params.layout_S),
                      EnableSoftmax, WithCausal, KernelTraits>
          <<<grid_dims, block_dim>>>(
              params.layout_Q, params.tme_load_Q,
              params.layout_K, params.key_desc, params.ptr_K,
              params.layout_V, params.tme_load_V,
              params.layout_O, params.ptr_O,
              params.layout_S, params.ptr_S,
              params.n_tiles, params.rln2_scale);
      error = musaGetLastError();
      if (error != musaSuccess) {
        std::cerr << "Launch FA failed. Error is " << musaGetErrorString(error) << std::endl;
        return false;
      }
    }

    double time = timer.seconds() / float(options.benchmark_iter);
    double gflops = double(2) * options.batch_size * (double(options.dim_sizeqk) * options.q_seq_len * options.kv_seq_len + double(options.dim_sizev) * options.q_seq_len * options.kv_seq_len) / double(1e9);

    double perf = gflops / time;
    printf("FA Fwd Perf: [%6.1f] GFLOPS,  (%6.4f) ms\n", perf, time * 1000.0f);

    return true;
  }
};


int main(int argc, char const** argv) {
  musaDeviceProp props;
  MUSA_CHECK(musaGetDeviceProperties(&props, 0));

  if (props.major * 10 + props.minor != 31) {
    std::cout
      << "This experimental code requires a GPU of MooreThreads's MP31 Architecture.\n";
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  constexpr int TileM = 32;
  constexpr int TileN = 64;
  constexpr int HeadDimQK = 576;
  constexpr int HeadDimV = 512;
  constexpr int KStages = 1;
  constexpr int VStages = 1;

  {
    using KernelTraits = FlashAttentionFwdKernelTraits<TileM, TileN, HeadDimQK, HeadDimV, mutlass::bfloat16_t, KStages, VStages>;
    FlashAttentionFwdRunner<KernelTraits> runner;
    bool status = runner.run<true>(options);
    if (!status) {
      return 1;
    }
  }
  {
    using KernelTraits = FlashAttentionFwdKernelTraits<TileM, TileN, HeadDimQK, HeadDimV, mutlass::float_e4m3_t, KStages, VStages>;
    FlashAttentionFwdRunner<KernelTraits> runner;
    bool status = runner.run<true>(options);
    if (!status) {
      return 1;
    }
  }

  return 0;
}
