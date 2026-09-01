#include "collective/fmha_collective_tme_warpspecialized.hpp"
#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_fusion.hpp"
#include "kernel/fmha_kernel_tme_warpspecialzed.hpp"
#include "kernel/fmha_tile_scheduler.hpp"

#include "fmha_options.hpp"

#include <mutlass/device_kernel.h>
#include <mutlass/util/device_memory.h>
#include <mutlass/util/command_line.h>
#include <mutlass/util/GPU_Clock.hpp>

#include <helper.h>

#include <vector>
#include <random>

using namespace mute;

struct Options {
  bool help = false;
  bool verify = false;
  bool verbose = false;
  bool perf = false;

  int loop = 10;

  int b = 2;
  int q = 512;
  int k = 512;
  int h = 8;
  int h_k = 4;

  int d_qk = 192;
  int d_vo = 128;

  bool causal = false;
  bool varlen = false;

  void parse(int argc, char const** args) {
    mutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("verify")) {
      verify = true;
    }

    if (cmd.check_cmd_line_flag("causal")) {
      causal = true;
    }

    if (cmd.check_cmd_line_flag("varlen")) {
      varlen = true;
    }

    if (cmd.check_cmd_line_flag("verbose")) {
      verbose = true;
    }

    if (cmd.check_cmd_line_flag("perf")) {
      perf = true;
    }

    cmd.get_cmd_line_argument("b", b);
    cmd.get_cmd_line_argument("d_qk", d_qk);
    cmd.get_cmd_line_argument("d_vo", d_vo);
    cmd.get_cmd_line_argument("q", q);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("h", h);
    cmd.get_cmd_line_argument("h_k", h_k);

    cmd.get_cmd_line_argument("loop", loop);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Fmha Prefill forward\n"
        << "Options:\n"
        << "  --help          If specified, displays this usage statement.\n"
        << "  --verify        If specified, check the results.\n"
        << "  --causal        If specified, do the causal mask fusion.\n"
        << "  --varlen        If specified, enable variable seqlen length.\n"
        << "  --verbose       If specified, display more kernel information.\n"
        << "  --perf          If specified, run perf test.\n"
        << "  --b=<int>       BatchSize.\n"
        << "  --q=<int>       Query Seqlen.\n"
        << "  --k=<int>       Key&Value Seqlen.\n"
        << "  --h=<int>       Query Heads.\n"
        << "  --h_k=<int>     Key&Value Heads.\n"
        << "  --loop=<int>    Perf loop iterations\n"
        << "  --d_qk=<int>    HeadDim QK.\n"
        << "  --d_vo=<int>    HeadDim VO.\n"
        << std::endl;

    return out;
  }
};



template <
  bool kIsCausal,
  bool kIsVarlen,
  int Consumers,
  int CTA_KV,
  int HEADDIM_QK,
  int HEADDIM_VO
>
struct FmhaFwdRunner {
  using Element = mutlass::half_t;
  using ElementO = mutlass::half_t;

  using TensorStride = Stride<int, _1, Stride<int, int>>;

  static constexpr bool UpperLeft = false;

  // CTA_Q, CTA_KV, D_QK, D_VO
  using TileShape = Shape<Int<64*Consumers>, Int<CTA_KV>, Int<HEADDIM_QK>, Int<HEADDIM_VO>>;

  using Fusion = std::conditional_t<kIsCausal, mutlass::fmha::collective::CausalFusion<UpperLeft, false>,
                                               mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaMainloopTmeWarpSpecialized<
    Element, float,
    TileShape,
    TensorStride, TensorStride, TensorStride,
    Fusion,
    mutlass::fmha::Option<mutlass::fmha::Tag::NumMmaWarpSquads, Int<Consumers>>,
    mutlass::fmha::Option<mutlass::fmha::Tag::Varlen, conditional_t<kIsVarlen, true_type, false_type>>,
    mutlass::fmha::Option<mutlass::fmha::Tag::KStage, Int<1>>,
    mutlass::fmha::Option<mutlass::fmha::Tag::VStage, Int<1>>>;


  using EpilogueTileShape = Shape<Int<tuple_element_t<0, TileShape>{} / Consumers>, tuple_element_t<3, TileShape>>;

  using CollectiveEpilogue = mutlass::fmha::collective::FmhaFwdEpilogue<
    ElementO, float, EpilogueTileShape,
    TensorStride, Stride<_1, Stride<int, int>>>;

  using TileScheduler = mutlass::fmha::kernel::FmhaIndividualTileScheduler<kIsCausal>;

  using FmhaKernel = mutlass::fmha::kernel::FmhaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  using ProblemShapeType = typename FmhaKernel::ProblemShape;

  // Member
  std::vector<int> cumulative_seqlen_q;
  std::vector<int> cumulative_seqlen_k;

  TensorStride stride_Q;
  TensorStride stride_K;
  TensorStride stride_V;
  TensorStride stride_O;
  Stride<_1, Stride<int, int>> stride_LSE;

  mutlass::DeviceAllocation<Element> block_Q;
  mutlass::DeviceAllocation<Element> block_K;
  mutlass::DeviceAllocation<Element> block_V;
  mutlass::DeviceAllocation<ElementO> block_O;
  mutlass::DeviceAllocation<float> block_LSE;

  mutlass::DeviceAllocation<int> device_cumulative_seqlen_q;
  mutlass::DeviceAllocation<int> device_cumulative_seqlen_k;

  std::vector<Element> data_Q;
  std::vector<Element> data_K;
  std::vector<Element> data_V;

  template <class ProblemShape>
  auto initialize_varlen(
      const Options& options, const ProblemShape& problem_size,
      bool same_varlen = true) {

    std::mt19937 rng(1234);
    std::normal_distribution<double> dist_q(get<0>(problem_size), get<0>(problem_size) / 2);
    std::normal_distribution<double> dist_k(get<1>(problem_size), get<1>(problem_size) / 2);

    int B = back(problem_size);

		auto generate_positive_int = [](auto& dist, auto& gen) {
      int result = 0;
      do {
        result = static_cast<int>(dist(gen));
      } while (result <= 0);
      return result;
    };

    cumulative_seqlen_q = {0};
    cumulative_seqlen_k = {0};

    int total_seq_q = 0;
    int total_seq_k = 0;
    int max_seq_q = 0;
    int max_seq_k = 0;

    for (int i = 0; i < B; ++i) {
      int seqlen_q = same_varlen ? get<0>(problem_size) : generate_positive_int(dist_q, rng);
      int seqlen_k = same_varlen ? get<1>(problem_size) : generate_positive_int(dist_k, rng);

      total_seq_q += seqlen_q;
      total_seq_k += seqlen_k;

      max_seq_q = std::max(max_seq_q, seqlen_q);
      max_seq_k = std::max(max_seq_k, seqlen_k);

      cumulative_seqlen_q.push_back(cumulative_seqlen_q.back() + seqlen_q);
      cumulative_seqlen_k.push_back(cumulative_seqlen_k.back() + seqlen_k);
    }

    ProblemShape problem_size_for_init = problem_size;
    get<0>(problem_size_for_init) = total_seq_q;
    get<1>(problem_size_for_init) = total_seq_k;
    get<6>(problem_size_for_init) = 1;

    ProblemShapeType problem_size_for_launch;
    get<0>(problem_size_for_launch) = mutlass::fmha::collective::VariableLength{total_seq_q, max_seq_q};
    get<1>(problem_size_for_launch) = mutlass::fmha::collective::VariableLength{total_seq_k, max_seq_k};
    get<2>(problem_size_for_launch) = get<2>(problem_size);
    get<3>(problem_size_for_launch) = get<3>(problem_size);
    get<4>(problem_size_for_launch) = get<4>(problem_size);
    get<5>(problem_size_for_launch) = get<5>(problem_size);
    get<6>(problem_size_for_launch) = get<6>(problem_size);

    return make_tuple(problem_size_for_init, problem_size_for_launch);
  }

  ProblemShapeType initialize(const Options& options) {
    int h_r = options.h / options.h_k;

    auto problem_shape_in = make_shape(options.q, options.k, options.d_qk, options.d_vo, options.h, options.h_k, options.b);
    ProblemShapeType problem_shape;
    decltype(problem_shape_in) problem_size;

    if constexpr (kIsVarlen) {
      auto [problem_shape_init, problem_shape_launch] = initialize_varlen(options, problem_shape_in);
      problem_shape = problem_shape_launch;
      problem_size = problem_shape_init;
    }
    else {
      problem_shape = problem_shape_in;
      problem_size = problem_shape_in;
    }

    // Init data
    auto [Q, K, D_QK, D_VO, H, H_K, B] = problem_size;

    auto layout_Q = make_ordered_layout(make_shape(Q, D_QK, make_shape(H, B)),   Step<_2, _0, Step<_1, _3>>{});
    auto layout_K = make_ordered_layout(make_shape(K, D_QK, make_shape(H_K, B)), Step<_2, _0, Step<_1, _3>>{});
    auto layout_V = make_ordered_layout(make_shape(K, D_VO, make_shape(H_K, B)), Step<_2, _0, Step<_1, _3>>{});
    auto layout_O = make_ordered_layout(make_shape(Q, D_VO, make_shape(H, B)),   Step<_2, _0, Step<_1, _3>>{});
    auto layout_LSE = make_ordered_layout(make_shape(Q, make_shape(H, B)), Step<_0, Step<_1, _2>>{});

    stride_Q = layout_Q.stride();
    stride_K = layout_K.stride();
    stride_V = layout_V.stride();
    stride_O = layout_O.stride();
    stride_LSE = layout_LSE.stride();

    block_Q.reset(cosize(layout_Q));
    block_K.reset(cosize(layout_K));
    block_V.reset(cosize(layout_V));
    block_O.reset(cosize(layout_O));
    block_LSE.reset(cosize(layout_LSE));

    data_Q.resize(cosize(layout_Q));
    data_K.resize(cosize(layout_K));
    data_V.resize(cosize(layout_V));

    for (int i = 0; i < data_Q.size(); ++i) {
      data_Q[i] = Element(float(rand() % 3 - 2));
    }

    for (int i = 0; i < data_K.size(); ++i) {
      data_K[i] = Element(float(rand() % 3 - 2));
    }

    for (int i = 0; i < data_V.size(); ++i) {
      data_V[i] = Element(float(rand() % 3 - 2));
    }

    block_Q.copy_from_host(data_Q.data());
    block_K.copy_from_host(data_K.data());
    block_V.copy_from_host(data_V.data());

    if constexpr (kIsVarlen) {
      device_cumulative_seqlen_q.reset(cumulative_seqlen_q.size());
      device_cumulative_seqlen_k.reset(cumulative_seqlen_k.size());

      device_cumulative_seqlen_q.copy_from_host(cumulative_seqlen_q.data());
      device_cumulative_seqlen_k.copy_from_host(cumulative_seqlen_k.data());

      get<0>(problem_shape).cumulative_length = device_cumulative_seqlen_q.get();
      get<1>(problem_shape).cumulative_length = device_cumulative_seqlen_k.get();
    }

    return problem_shape;
  }

  template <typename ElementCheck>
  bool verify_tensor(std::vector<ElementCheck> vector_Input,
                     std::vector<ElementCheck> vector_Input_Ref,
                     bool printValues = false, bool printDiffs = false,
                     float errCountExpected = 0, int64_t verify_length = -1) {
    int64_t size = (vector_Input.size() < vector_Input_Ref.size())
        ? vector_Input.size()
        : vector_Input_Ref.size();
    size = (verify_length == -1) ? size : verify_length;

    float abs_tol = 5e-3f;

    float rel_tol = 5e-3f;
    int errCount = 0;
    for (int64_t i = 0; i < size; ++i) {
      if (printValues)
        std::cout << vector_Input[i] << " " << vector_Input_Ref[i] << std::endl;
      float diff = (float)(vector_Input[i] - vector_Input_Ref[i]);
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref[i] + 1e-5f);
      float relative_diff = abs_diff / abs_ref;
      bool both_inf = std::isinf(vector_Input[i]) && std::isinf(vector_Input_Ref[i]);

      if (!both_inf && ((std::isnan(vector_Input_Ref[i]) || std::isnan(abs_diff) || std::isinf(abs_diff)) ||
          (abs_diff > abs_tol && relative_diff > rel_tol))) {
        if (printDiffs)
          printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n",
                 int(i), int(size), abs_diff, relative_diff,
                 (float)(vector_Input[i]), (float)(vector_Input_Ref[i]));
        errCount++;
        return false;
      }
    }
    auto errCountComputed = float(errCount) / float(size) * 100;

    return errCountComputed <= errCountExpected ? true : false;
  }

  template <class Params>
  bool verify(ProblemShapeType problem_shape, Params const& params) {
    auto [Q, K, D_QK, D_VO, H, H_K, B] = problem_shape;
    int H_R = H / H_K;

    int total_seq_q = 0;
    int total_seq_k = 0;

    if constexpr (kIsVarlen) {
      total_seq_q = Q.total_length;
      total_seq_k = K.total_length;
    } else {
      total_seq_q = Q;
      total_seq_k = K;
    }

    Tensor mQ = make_tensor(data_Q.data(), make_shape(total_seq_q, D_QK, make_shape(H, B)), stride_Q);
    Tensor mK = make_tensor(data_K.data(), make_shape(total_seq_k, D_QK, make_shape(H_K, B)), stride_K);
    Tensor mV = make_tensor(data_V.data(), make_shape(total_seq_k, D_VO, make_shape(H_K, B)), stride_V);

    std::vector<ElementO> ref_O(block_O.capacity);
    std::vector<float> ref_LSE(block_LSE.capacity);

    Tensor mRefO = make_tensor(ref_O.data(), make_shape(total_seq_q, D_VO, make_shape(H, B)), stride_O);
    Tensor mRefLSE = make_tensor(ref_LSE.data(), make_shape(total_seq_q, make_shape(H, B)), stride_LSE);

    for (int b = 0; b < B; ++b) {
      int seqlen_q = 0;
      int seqlen_k = 0;
      int prefix_q = 0;
      int prefix_k = 0;
      int bs_id = b;

      if constexpr (kIsVarlen) {
        seqlen_q = cumulative_seqlen_q[b+1] - cumulative_seqlen_q[b];
        seqlen_k = cumulative_seqlen_k[b+1] - cumulative_seqlen_k[b];
        prefix_q = cumulative_seqlen_q[b];
        prefix_k = cumulative_seqlen_k[b];
        bs_id = 0;
      } else {
        seqlen_q = Q;
        seqlen_k = K;
      }

      for (int h_id = 0; h_id < H; ++h_id) {
        int h_k_id = h_id / H_R;

        std::vector<float> acc_qk(seqlen_q * seqlen_k);

        Tensor mP = make_tensor(acc_qk.data(), make_layout(make_shape(seqlen_q, seqlen_k), LayoutRight{}));

        // QK
        for (int q_id = 0; q_id < seqlen_q; ++q_id) {
          for (int k_id = 0; k_id < seqlen_k; ++k_id) {
            float acc = 0.f;
            for (int d = 0; d < D_QK; ++d) {
              acc += float(mQ(prefix_q + q_id, d, make_coord(h_id, bs_id))) * float(mK(prefix_k + k_id, d, make_coord(h_k_id, bs_id)));
            }
            mP(q_id, k_id) = acc;

            if (kIsCausal) {
              bool need_mask = false;

              if (UpperLeft) {
                need_mask = q_id < k_id;
              } else {
                need_mask = q_id + (seqlen_k - seqlen_q) < k_id;
              }
              if (need_mask) {
                mP(q_id, k_id) = -std::numeric_limits<float>::infinity();
              }
            }
          }
        }

        // Softmax
        std::vector<float> row_sum(seqlen_q);
        std::vector<float> row_max(seqlen_q);

        for (int q_id = 0; q_id < seqlen_q; ++q_id) {
          row_max[q_id] = mP(q_id, 0);
          for (int k_id = 1; k_id < seqlen_k; ++k_id) {
            row_max[q_id] = std::max(row_max[q_id], mP(q_id, k_id));
          }

          float local_max = row_max[q_id] == -std::numeric_limits<float>::infinity() ? 0.0f : row_max[q_id];
          for (int k_id = 0; k_id < seqlen_k; ++k_id) {
            mP(q_id, k_id) = std::exp2f(mP(q_id, k_id) * params.mainloop.sm_scale_log2 - local_max * params.mainloop.sm_scale_log2);
            row_sum[q_id] += mP(q_id, k_id);
          }
        }

        // PV
        for (int q_id = 0; q_id < seqlen_q; ++q_id) {
          mRefLSE(prefix_q + q_id, make_coord(h_id, bs_id)) = (row_sum[q_id] == 0.f || row_sum[q_id] != row_sum[q_id] ) ?
                                                -std::numeric_limits<float>::infinity()
                                                : std::logf(row_sum[q_id]) + row_max[q_id] * params.mainloop.sm_scale;

          float inv_sum = (row_sum[q_id] == 0.f || row_sum[q_id] != row_sum[q_id]) ? 0.0f : 1.0f / row_sum[q_id];

          for (int d = 0; d < D_VO; ++d) {
            float acc = 0.f;
            for (int k_id = 0; k_id < seqlen_k; ++k_id) {
              acc += float(mP(q_id, k_id)) * float(mV(prefix_k + k_id, d, make_coord(h_k_id, bs_id)));
            }
            mRefO(prefix_q + q_id, d, make_coord(h_id, bs_id)) = acc * inv_sum;
          }
        }
      }
    }

    std::vector<ElementO> gpu_O(block_O.capacity);
    std::vector<float> gpu_LSE(block_LSE.capacity);

    block_O.copy_to_host(gpu_O.data());
    block_LSE.copy_to_host(gpu_LSE.data());

    bool output_passed = verify_tensor(gpu_O, ref_O, false, true);
    if (!output_passed) {
      printf("output check failed\n");
    }

    bool lse_passed = verify_tensor(gpu_LSE, ref_LSE, false, true);
    if (!lse_passed) {
      printf("lse check failed\n");
    }

    return output_passed && lse_passed;

  }

  int run(const Options& options) {
    ProblemShapeType problem_shape = initialize(options);

    typename FmhaKernel::Arguments arguments {
      problem_shape,
      {
        block_Q.get(), stride_Q,
        block_K.get(), stride_K,
        block_V.get(), stride_V,
        1.0f / sqrtf(get<2>(problem_shape)),
      },
      {
        block_O.get(), stride_O,
        block_LSE.get(), stride_LSE
      },
    };

    typename FmhaKernel::Params params = FmhaKernel::to_underlying_arguments(arguments);

    auto grid_dim = TileScheduler::get_grid_shape(params.scheduler);

    mutlass::device_kernel<FmhaKernel><<<grid_dim, FmhaKernel::MaxThreadsPerBlock, FmhaKernel::SharedStorageSize>>>(params);

    musaError_t result = musaDeviceSynchronize();

    if (result != musaSuccess) {
      printf("error when enqueue kerenl:%d (%s)\n", int(result), musaGetErrorString(result));
      return -1;
    }

    // verify
    if (options.verify) {
      bool passed = verify(problem_shape, params);

      if (passed) {
        printf("Reference check pass\n");
      } else {
        printf("Reference check failed\n");
        return -1;
      }
    }


    // Perf
    if (options.perf) {
      double flops = 0.0;
      if constexpr (kIsVarlen) {
        for (int i = 0; i < get<6>(problem_shape); ++i) {
          int seqlen_q = cumulative_seqlen_q[i+1] - cumulative_seqlen_q[i];
          int seqlen_k = cumulative_seqlen_k[i+1] - cumulative_seqlen_k[i];

          flops += 1.0 * seqlen_q * seqlen_k * (get<2>(problem_shape) + get<3>(problem_shape));
        }
      } else {
        flops += 1.0 * get<0>(problem_shape) * get<1>(problem_shape) * (get<2>(problem_shape) + get<3>(problem_shape)) * get<6>(problem_shape);
      }

      flops *= kIsCausal ? 0.5f : 1.0;
      flops *= 2.0 * get<4>(problem_shape);

      double gflops = flops / double(1e9);

      // warmup
      for (int i = 0; i < 1; ++i) {
        mutlass::device_kernel<FmhaKernel><<<grid_dim, FmhaKernel::MaxThreadsPerBlock, FmhaKernel::SharedStorageSize>>>(params);
      }


      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.loop; ++i) {
        mutlass::device_kernel<FmhaKernel><<<grid_dim, FmhaKernel::MaxThreadsPerBlock, FmhaKernel::SharedStorageSize>>>(params);
      }

      double time = timer.seconds() / float(options.loop);
      double perf = gflops / time;

      printf("Fmha Perf:[%6.1f] GFLOPS, (%6.4f) ms\n", perf, time * 1000.0f);
    }
    return 0;
  }
};

template <
  bool kIsCausal,
  bool kIsVarlen
>
int dispatch_varlen(const Options& options) {
  if (options.d_qk == 128 && options.d_vo == 128) {
    using Runner = FmhaFwdRunner<kIsCausal, kIsVarlen, 4, 128, 128, 128>;

    Runner runner;

    return runner.run(options);
  } else if (options.d_qk == 192 && options.d_vo == 128) {
    using Runner = FmhaFwdRunner<kIsCausal, kIsVarlen, 4, 64, 192, 128>;

    Runner runner;

    return runner.run(options);
  } else {
    printf("Unsupported d_qk:%d, d_vo:%d\n", options.d_qk, options.d_vo);
    return -1;
  }
}

template <bool kIsCausal>
int dispatch_causal(const Options& options) {
  if (options.varlen) {
    return dispatch_varlen<kIsCausal, true>(options);
  } else {
    return dispatch_varlen<kIsCausal, false>(options);
  }
}

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

  if (options.causal) {
    return dispatch_causal<true>(options);
  } else {
    return dispatch_causal<false>(options);
  }
}
