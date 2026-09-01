#include "collective/fmha_paged_collective_tme_warpspecialized.hpp"
#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_fusion.hpp"
#include "kernel/fmha_paged_kernel_tme_warpspecialzed.hpp"
#include "kernel/fmha_tile_scheduler.hpp"

#include "fmha_options.hpp"

#include <mutlass/device_kernel.h>
#include <mutlass/util/device_memory.h>
#include <mutlass/util/command_line.h>
#include <mutlass/util/GPU_Clock.hpp>

#include <helper.h>

#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

using namespace mute;

struct Options {
  bool help = false;
  bool verify = false;
  bool verbose = false;
  bool perf = false;

  int loop = 10;

  int b = 1;
  int q = 1;
  int k = 512;
  int h = 1;
  int h_k = 1;

  int page_size = 64;
  float spread = 0.0f;

  int d_qk = 128;
  int d_vo = 128;
  bool causal = false;

  void parse(int argc, char const** args) {
    mutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("verify")) {
      verify = true;
    }

    if (cmd.check_cmd_line_flag("verbose")) {
      verbose = true;
    }

    if (cmd.check_cmd_line_flag("perf")) {
      perf = true;
    }

    cmd.get_cmd_line_argument("b", b);
    cmd.get_cmd_line_argument("q", q);
    cmd.get_cmd_line_argument("k", k);
    cmd.get_cmd_line_argument("page", page_size);
    cmd.get_cmd_line_argument("h", h);
    cmd.get_cmd_line_argument("h_k", h_k);
    cmd.get_cmd_line_argument("d_qk", d_qk);
    cmd.get_cmd_line_argument("d_vo", d_vo);

    cmd.get_cmd_line_argument("loop", loop);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "Paged Fmha forward\n"
        << "Options:\n"
        << "  --help          If specified, displays this usage statement.\n"
        << "  --verify        If specified, check the results.\n"
        << "  --causal        If specified, do the causal mask fusion.\n"
        << "  --verbose       If specified, display more kernel information.\n"
        << "  --perf          If specified, run perf test.\n"
        << "  --spread        Relative spread away from K (determine valen range).\n"
        << "  --b=<int>       BatchSize.\n"
        << "  --q=<int>       Query Seqlen.(default=1, > 1 for MTP)\n"
        << "  --k=<int>       Key&Value Seqlen.\n"
        << "  --h=<int>       Query Heads.\n"
        << "  --h_k=<int>     Key&Value Heads.\n"
        << "  --page=<int>    Set the page size.(default:64)\n"
        << "  --loop=<int>    Perf loop iterations\n"
        << std::endl;
    return out;
  }
};

template <bool kIsCausal>
struct PagedFmhaRunner {
  using Element = mutlass::half_t;
  using ElementO = mutlass::half_t;

  using TensorStride = Stride<int, _1, Stride<int, int>>;

  static constexpr int Consumers = 4;
  static constexpr bool UpperLeft = false;
  static constexpr bool PackGQA = false;

  // CTA_Q, CTA_KV, D_QK, D_VO
  using TileShape = Shape<Int<64*Consumers>, _64, _128, _128>;

  using Fusion = std::conditional_t<kIsCausal, mutlass::fmha::collective::CausalFusion<UpperLeft, PackGQA>,
                                               mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaPagedMainloopTmeWarpSpecialized<
    Element, float,
    TileShape,
    TensorStride, TensorStride, TensorStride,
    Fusion,
    mutlass::fmha::Option<mutlass::fmha::Tag::NumMmaWarpSquads, Int<Consumers>>,
    mutlass::fmha::Option<mutlass::fmha::Tag::KStage, Int<2>>,
    mutlass::fmha::Option<mutlass::fmha::Tag::VStage, Int<2>>>;

  using EpilogueTileShape = Shape<Int<tuple_element_t<0, TileShape>{} / Consumers>, tuple_element_t<3, TileShape>>;

  using CollectiveEpilogue = mutlass::fmha::collective::FmhaFwdEpilogue<
    ElementO, float, EpilogueTileShape,
    TensorStride, Stride<_1, Stride<int, int>>>;

  using TileScheduler = mutlass::fmha::kernel::FmhaIndividualTileScheduler<kIsCausal>;

  using FmhaKernel = mutlass::fmha::kernel::PagedFmhaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;

  using ProblemShapeType = typename FmhaKernel::ProblemShape;

  // Member
  TensorStride stride_Q;
  TensorStride stride_K;
  TensorStride stride_V;
  TensorStride stride_O;
  Stride<_1, Stride<int, int>> stride_LSE;
  Stride<int, _1> stride_PT;

  mutlass::DeviceAllocation<Element> block_Q;
  mutlass::DeviceAllocation<Element> block_K;
  mutlass::DeviceAllocation<Element> block_V;
  mutlass::DeviceAllocation<ElementO> block_O;
  mutlass::DeviceAllocation<float> block_LSE;
  mutlass::DeviceAllocation<int> block_PageTable;
  mutlass::DeviceAllocation<int> block_Seq;

  std::vector<Element> data_Q;
  std::vector<Element> data_K;
  std::vector<Element> data_V;
  std::vector<int> data_PageTable;
  std::vector<int> data_Seq;

  int page_count;
  int num_splits = 1;

  ProblemShapeType initialize(const Options& options) {
    int h_r = options.h / options.h_k;

    int max_K = static_cast<int>((1 + options.spread) * options.k);
    int min_K = static_cast<int>((1 - options.spread) * options.k);
    int page_size = options.page_size;
    page_count = options.b * ceil_div(max_K, page_size);

    ProblemShapeType problem_shape = make_shape(options.q, max_K, options.d_qk, options.d_vo, options.h, options.h_k, options.b);

    auto [Q, K, D_QK, D_VO, H, H_K, B] = problem_shape;

    auto layout_Q = make_ordered_layout(make_shape(Q, D_QK, make_shape(H, B)), Step<_2, _0, Step<_1, _3>>{});
    auto layout_K = make_ordered_layout(make_shape(page_size, D_QK, make_shape(H_K, page_count)), Step<_2, _0, Step<_1, _3>>{});
    auto layout_V = make_ordered_layout(make_shape(page_size, D_VO, make_shape(H_K, page_count)), Step<_2, _0, Step<_1, _3>>{});

    auto layout_O = make_ordered_layout(make_shape(Q, D_VO, make_shape(H, B)), Step<_2, _0, Step<_1, _3>>{});

    auto layout_LSE = make_ordered_layout(make_shape(Q, make_shape(H, B)), Step<_0, Step<_1, _2>>{});

    stride_Q = layout_Q.stride();
    stride_K = layout_K.stride();
    stride_V = layout_V.stride();
    stride_O = layout_O.stride();
    stride_LSE = layout_LSE.stride();
    stride_PT = make_stride(page_count, _1{});

    block_Q.reset(cosize(layout_Q));
    block_K.reset(cosize(layout_K));
    block_V.reset(cosize(layout_V));
    block_O.reset(cosize(layout_O));
    block_LSE.reset(cosize(layout_LSE));

    block_PageTable.reset(page_count * B);
    block_Seq.reset(B);

    // Init
    data_Q.resize(cosize(layout_Q));
    data_K.resize(cosize(layout_K));
    data_V.resize(cosize(layout_V));
    data_PageTable.resize(page_count * B);
    data_Seq.resize(B);


    for (int i = 0; i < data_Q.size(); ++i) {
      data_Q[i] = Element(float(rand() % 3 - 2));
    }

    for (int i = 0; i < data_K.size(); ++i) {
      data_K[i] = Element(float(rand() % 3 - 2));
    }

    for (int i = 0; i < data_V.size(); ++i) {
      data_V[i] = Element(float(rand() % 3 - 2));
    }

    for (int i = 0; i < B; ++i) {
      int seqlen = min_K + rand() % (max_K - min_K + 1);
      data_Seq[i] = seqlen;

      for (int j = 0; j < ceil_div(seqlen, page_size); ++j) {
        data_PageTable[i * page_count + j] = i + j * B;
      }
    }
    block_Q.copy_from_host(data_Q.data());
    block_K.copy_from_host(data_K.data());
    block_V.copy_from_host(data_V.data());
    block_PageTable.copy_from_host(data_PageTable.data());
    block_Seq.copy_from_host(data_Seq.data());

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
      if ((std::isnan(vector_Input_Ref[i]) || std::isnan(abs_diff) || std::isinf(abs_diff)) ||
          (abs_diff > abs_tol && relative_diff > rel_tol)) {
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
    auto [Q, K_, D_QK, D_VO, H, H_K, B] = problem_shape;

    int H_R = H / H_K;

    int PageSize = params.mainloop.page_size;
    int PageCount = params.mainloop.page_count;

    Tensor mQ = make_tensor(data_Q.data(), make_shape(Q, D_QK, make_shape(H, B)), stride_Q);
    Tensor mK = make_tensor(data_K.data(), make_shape(PageSize, D_QK, make_shape(H_K, PageCount)), stride_K);
    Tensor mV = make_tensor(data_V.data(), make_shape(PageSize, D_VO, make_shape(H_K, PageCount)), stride_V);

    Tensor mPT = make_tensor(data_PageTable.data(), make_shape(B, PageCount), stride_PT);
    Tensor mSeq = make_tensor(data_Seq.data(), make_shape(B));

    std::vector<ElementO> ref_O(block_O.capacity);
    std::vector<float> ref_LSE(block_LSE.capacity);

    Tensor mRefO = make_tensor(ref_O.data(), make_shape(Q, D_VO, make_shape(H, B)), stride_O);
    Tensor mRefLSE = make_tensor(ref_LSE.data(), make_shape(Q, make_shape(H, B)), stride_LSE);

    // TODO: we assume PageSize = BN for simplicity now...
    for (int b = 0; b < B; ++b) {
      for (int h_id = 0; h_id< H; ++h_id) {
        int h_k_id = h_id / H_R;

        int cur_seqlen = mSeq(b);

        int n_tiles = ceil_div(cur_seqlen, get<1>(TileShape{}));
        int n_tiles_per_split = ceil_div(n_tiles, num_splits);

        for (int split_id = 0; split_id < num_splits; ++split_id) {
          int split_kv_start = n_tiles_per_split * split_id;
          int split_kv_end = min(split_kv_start + n_tiles_per_split, n_tiles);

          if (split_kv_end <= split_kv_start) continue;

          int cur_split_tiles = split_kv_end - split_kv_start;

          std::vector<float> acc_qk(cur_split_tiles * get<1>(TileShape{}) * Q, -std::numeric_limits<float>::infinity());
          Tensor mP = make_tensor(acc_qk.data(), make_layout(make_shape(Q, cur_split_tiles * get<1>(TileShape{})), LayoutRight{}));

          // QK
          for (int q_id = 0; q_id < Q; ++q_id) {
            for (int k_block = split_kv_start, outer = 0; k_block < split_kv_end; ++k_block, ++outer) {
              int page_number = mPT(b, k_block);
              for (int inner = 0; inner < get<1>(TileShape{}); ++inner) {
                int token_id = k_block * get<1>(TileShape{}) + inner;

                if (token_id < cur_seqlen) {
                  float acc = 0.f;
                  for (int d = 0; d < D_QK; ++d) {
                    acc += float(mQ(q_id, d, make_shape(h_id, b))) * float(mK(inner, d, make_shape(h_k_id, page_number)));
                  }
                  mP(q_id, outer * get<1>(TileShape{}) + inner) = acc;


                  // Bottom Right
                  if (kIsCausal && (q_id + (cur_seqlen - Q) < token_id)) {
                    mP(q_id, outer * get<1>(TileShape{}) + inner) = -std::numeric_limits<float>::infinity();
                  }
                }
              }
            }
          }

          // Softmax
          std::vector<float> row_sum(Q);
          std::vector<float> row_max(Q);

          for (int q_id = 0; q_id < Q; ++q_id) {
            row_max[q_id] = mP(q_id, 0);
            for (int k_id = 1; k_id < cur_split_tiles * get<1>(TileShape{}); ++k_id) {
              row_max[q_id] = std::max(row_max[q_id], mP(q_id, k_id));
            }

            float local_max = row_max[q_id] == -std::numeric_limits<float>::infinity() ? 0.0f : row_max[q_id];
            for (int k_id = 0; k_id < cur_split_tiles * get<1>(TileShape{}); ++k_id) {
              mP(q_id, k_id) = std::exp2f(mP(q_id, k_id) * params.mainloop.sm_scale_log2 - local_max * params.mainloop.sm_scale_log2);
              row_sum[q_id] += mP(q_id, k_id);
            }
          }

          // PV
          for (int q_id = 0; q_id < Q; ++q_id) {
            mRefLSE(q_id, make_shape(h_id, b)) = (row_sum[q_id] == 0.f || row_sum[q_id] != row_sum[q_id]) ? -std::numeric_limits<float>::infinity() : std::logf(row_sum[q_id]) + row_max[q_id] * params.mainloop.sm_scale;
            float inv_sum = (row_sum[q_id] == 0.f || row_sum[q_id] != row_sum[q_id]) ? 0.0f : 1.0f / row_sum[q_id];
            for (int d = 0; d < D_VO; ++d) {
              float acc = 0.0f;
              for (int k_block = split_kv_start, outer = 0; k_block < split_kv_end; ++k_block, ++outer) {
                int page_number = mPT(b, k_block);

                for (int inner = 0; inner < get<1>(TileShape{}); ++inner) {
                  int token_id = k_block * get<1>(TileShape{}) + inner;

                  if (token_id < cur_seqlen) {
                    acc += float(mP(q_id, outer * get<1>(TileShape{}) + inner)) * float(mV(inner, d, make_shape(h_k_id, page_number)));
                  }
                }
              }
              mRefO(q_id, d, make_shape(h_id, b)) = acc * inv_sum;
            }
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
        block_PageTable.get(), stride_PT,
        block_Seq.get(),

        1.0f / sqrtf(get<2>(problem_shape)),
        options.page_size,
        page_count,
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
      auto [Q, K_, D_QK, D_VO, H, H_K, B] = problem_shape;
      for (int i = 0; i < B; ++i) {
        int K = data_Seq[i];
        flops += 1.0f * Q * K * (D_QK + D_VO);
      }
      flops *= kIsCausal ? 0.5f : 1.0f;
      flops *= 2.0 * H;

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

template <bool kIsCausal>
int dispatch_causal(const Options& options) {
  using Runner = PagedFmhaRunner<kIsCausal>;
  Runner runner;
  return runner.run(options);
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
