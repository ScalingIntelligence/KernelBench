#include "collective/fmha_collective_epilogue.hpp"
#include "collective/fmha_fusion.hpp"
#include "collective/fmha_mla_collective_tme_warpspecialized.hpp"
#include "kernel/mla_kernel_tme_warpspecialzed.hpp"


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
  int tp = 1;

  int page_size = 64;
  float spread = 0.0f;

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
    cmd.get_cmd_line_argument("tp", tp);

    cmd.get_cmd_line_argument("loop", loop);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out << "MLA Decode forward\n"
        << "Options:\n"
        << "  --help          If specified, displays this usage statement.\n"
        << "  --verify        If specified, check the results.\n"
        << "  --verbose       If specified, display more kernel information.\n"
        << "  --perf          If specified, run perf test.\n"
        << "  --spread        Relative spread away from K (determine valen range).\n"
        << "  --b=<int>       BatchSize.\n"
        << "  --q=<int>       Query Seqlen.(default=1, > 1 for MTP)\n"
        << "  --k=<int>       Key&Value Seqlen.\n"
        << "  --tp=<int>      TesnorParallelism number (head = 128/tp).\n"
        << "  --page=<int>    Set the page size.(default:64)\n"
        << "  --loop=<int>    Perf loop iterations\n"
        << std::endl;
    return out;
  }
};

struct MlaRunner {
  using Element = mutlass::half_t;
  using ElementO = mutlass::half_t;

  // Q*H, D, (1, B)
  using StrideQ = Stride<int, _1, Stride<_0, int>>;
  // PageSize, D, (1, PageCount)
  using StrideC = Stride<int, _1, Stride<_0, int>>;

  // CTA_Q, CTA_K, (Latent, Rope)
  using TileShape = Shape<_32, _64, Shape<_512, _64>>;

  static constexpr bool IsCausal = true;
  // BottomRight + PackGQA causal
  using Fusion = conditional_t<IsCausal, mutlass::fmha::collective::CausalFusion<false, true>,
                                         mutlass::fmha::collective::DefaultFusion>;

  using CollectiveMainloop = mutlass::fmha::collective::FmhaMlaMainloopTmeWarpSpecializedV2<
    Element, float,
    TileShape,
    StrideQ, StrideC,
    Fusion>;

  using CollectiveEpilogue = mutlass::fmha::collective::FmhaFwdEpilogue<
    Element, float, Shape<_32, _512>,
    StrideQ, Stride<_1, Stride<int, int>>>;

  using MlaKernel = mutlass::fmha::kernel::MlaKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue>;

  using ProblemShapeType = typename MlaKernel::ProblemShape;

  // Member
  StrideQ stride_Q_latent;
  StrideQ stride_Q_rope;
  StrideC stride_C_latent;
  StrideC stride_K_rope;
  Stride<int, _1> stride_PT;

  StrideC stride_O;
  Stride<_1, Stride<int, int>> stride_LSE;

  mutlass::DeviceAllocation<Element> block_Q;
  mutlass::DeviceAllocation<Element> block_C;
  mutlass::DeviceAllocation<int> block_PageTable;
  mutlass::DeviceAllocation<int> block_Seq;

  mutlass::DeviceAllocation<ElementO> block_O;
  mutlass::DeviceAllocation<float> block_LSE;

  std::vector<Element> data_Q;
  std::vector<Element> data_C;
  std::vector<int> data_PageTable;
  std::vector<int> data_Seq;

  int num_splits = 1;

  ProblemShapeType initialize(const Options& options) {
    int max_K = static_cast<int>((1 + options.spread) * options.k);
    int min_K = static_cast<int>((1 - options.spread) * options.k);
    int page_size = options.page_size;
    int page_count = options.b * ceil_div(max_K, page_size);
    auto [D_latent, D_rope] = get<2>(TileShape{});

    ProblemShapeType problem_shape = make_shape(options.q, page_size, make_shape(D_latent, D_rope), 128 / options.tp, page_count, options.b);

    auto [Q, PageSize, D, H, PageCount, B] = problem_shape;

    // Q*H, D, (1, B)
    stride_Q_latent = make_stride(D_latent + D_rope, _1{}, make_shape(_0{}, Q * H * (D_latent + D_rope)));
    stride_Q_rope = stride_Q_latent;

    // PageSize, D, (1, PageCount)
    stride_C_latent = make_stride(D_latent + D_rope, _1{}, make_stride(_0{}, PageSize * (D_latent + D_rope)));
    stride_K_rope = stride_C_latent;

    // PageTable
    stride_PT = make_stride(PageCount, _1{});

    // Ouput
    stride_O = make_stride(D_latent, _1{}, make_stride(_0{}, Q * H * D_latent));
    stride_LSE = make_stride(_1{}, make_stride(Q, Q * H));

    block_Q.reset(Q * H * B * (D_latent + D_rope));
    block_C.reset(PageSize * PageCount * (D_latent + D_rope));
    block_PageTable.reset(PageCount * B);
    block_Seq.reset(B);
    block_O.reset(Q * H * B * D_latent);
    block_LSE.reset(Q * H * B);

    // init
    data_PageTable.resize(PageCount * B);
    data_Seq.resize(B);

    for (int i = 0; i < B; ++i) {
      int seqlen = min_K + rand() % (max_K - min_K + 1);
      data_Seq[i] = seqlen;

      for (int j = 0; j < ceil_div(seqlen, PageSize); ++j) {
        data_PageTable[i * PageCount + j] = i + j * B;
      }
    }

    data_Q.resize(block_Q.capacity);
    data_C.resize(block_C.capacity);

    for (int i = 0; i < data_Q.size(); ++i) {
      data_Q[i] = Element(rand() % 3 - 2);
    }

    for (int i = 0; i < data_C.size(); ++i) {
      data_C[i] = Element(rand() % 3 - 2);
    }

    block_Q.copy_from_host(data_Q.data());
    block_C.copy_from_host(data_C.data());
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
    auto [Q, PageSize, D, H, PageCount, B] = problem_shape;
    auto [D_latent, D_rope] = D;
    int H_K = 1;

    int H_R = H / H_K;

    Tensor mQ_latent = make_tensor(data_Q.data(), make_shape(Q * H, D_latent, make_shape(1, B)), stride_Q_latent);
    Tensor mQ_rope = make_tensor(data_Q.data() + D_latent, make_shape(Q*H, D_rope, make_shape(1, B)), stride_Q_rope);

    Tensor mC_latent = make_tensor(data_C.data(), make_shape(PageSize, D_latent, make_shape(H_K, PageCount)), stride_C_latent);
    Tensor mK_rope = make_tensor(data_C.data() + D_latent, make_shape(PageSize, D_rope, make_shape(H_K, PageCount)), stride_K_rope);

    Tensor mPT = make_tensor(data_PageTable.data(), make_shape(B, PageCount), stride_PT);
    Tensor mSeq = make_tensor(data_Seq.data(), make_shape(B));

    std::vector<ElementO> ref_O(block_O.capacity);
    std::vector<float> ref_LSE(block_LSE.capacity);

    Tensor mRefO = make_tensor(ref_O.data(), make_shape(Q*H, D_latent, make_shape(1, B)), stride_O);
    Tensor mRefLSE = make_tensor(ref_LSE.data(), make_shape(Q*H, make_shape(1, B)), stride_LSE);

    // TODO: we assume PageSize = BN for simplicity now...
    for (int b = 0; b < B; ++b) {
      int cur_seqlen = mSeq(b);

      int n_tiles = ceil_div(cur_seqlen, get<1>(TileShape{}));
      int n_tiles_per_split = ceil_div(n_tiles, num_splits);

      for (int split_id = 0; split_id < num_splits; ++split_id) {
        int split_kv_start = n_tiles_per_split * split_id;
        int split_kv_end = min(split_kv_start + n_tiles_per_split, n_tiles);

        if (split_kv_end <= split_kv_start) continue;

        // since MLA H_K=1, just skip this loop

        int cur_split_tiles = split_kv_end - split_kv_start;

        std::vector<float> acc_qk(cur_split_tiles * get<1>(TileShape{}) * Q * H, -std::numeric_limits<float>::infinity());
        Tensor mP = make_tensor(acc_qk.data(), make_layout(make_shape(Q * H, cur_split_tiles * get<1>(TileShape{})), LayoutRight{}));

        // QK
        for (int qh_id = 0; qh_id < Q * H; ++qh_id) {
          for (int k_block = split_kv_start, outer = 0; k_block < split_kv_end; ++k_block, ++outer) {
            int page_number = mPT(b, k_block);

            for (int inner = 0; inner < get<1>(TileShape{}); ++inner) {
              int token_id = k_block * get<1>(TileShape{}) + inner;

              if (token_id < cur_seqlen) {
                float acc = 0.f;
                for (int d = 0; d < D_latent; ++d) {
                  acc += float(mQ_latent(qh_id, d, b)) * float(mC_latent(inner, d, page_number));
                }

                for (int d = 0; d < D_rope; ++d) {
                  acc += float(mQ_rope(qh_id, d, b)) * float(mK_rope(inner, d, page_number));
                }
                mP(qh_id, outer * get<1>(TileShape{}) + inner) = acc;

                // BottomRight + PackGQA
                int q_id = qh_id / H_R;
                if (q_id + (cur_seqlen - Q) < token_id) {
                  mP(qh_id, outer * get<1>(TileShape{}) + inner) = -std::numeric_limits<float>::infinity();
                }
              }
            }
          }
        }

        // Softmax
        std::vector<float> row_sum(Q * H);
        std::vector<float> row_max(Q * H);

        for (int qh_id = 0; qh_id < Q * H; ++qh_id) {
          row_max[qh_id] = mP(qh_id, 0);
          for (int k_id = 1; k_id < cur_split_tiles * get<1>(TileShape{}); ++k_id) {
            row_max[qh_id] = std::max(row_max[qh_id] , mP(qh_id, k_id));
          }

          float local_max = row_max[qh_id] == -std::numeric_limits<float>::infinity() ? 0.0f : row_max[qh_id];
          for (int k_id = 0; k_id < cur_split_tiles * get<1>(TileShape{}); ++k_id) {
            mP(qh_id, k_id) = std::exp2f(mP(qh_id, k_id) * params.mainloop.sm_scale_log2 - local_max * params.mainloop.sm_scale_log2);
            row_sum[qh_id] += mP(qh_id, k_id);
          }
        }

        // PV
        for (int qh_id = 0; qh_id < Q * H; ++qh_id) {
          mRefLSE(qh_id, b) = (row_sum[qh_id] == 0.f || row_sum[qh_id] != row_sum[qh_id]) ? -std::numeric_limits<float>::infinity() : std::logf(row_sum[qh_id]) + row_max[qh_id] * params.mainloop.sm_scale;
          float inv_sum = (row_sum[qh_id] == 0.f || row_sum[qh_id] != row_sum[qh_id]) ? 0.0f : 1.0f / row_sum[qh_id];
          for (int d = 0; d < D_latent; ++d) {
            float acc = 0.f;
            for (int k_block = split_kv_start, outer = 0; k_block < split_kv_end; ++k_block, ++outer) {
              int page_number = mPT(b, k_block);

              for (int inner = 0; inner < get<1>(TileShape{}); ++inner) {
                int token_id = k_block * get<1>(TileShape{}) + inner;

                if (token_id < cur_seqlen) {
                  acc += float(mP(qh_id, outer * get<1>(TileShape{}) + inner)) * float(mC_latent(inner, d, page_number));
                }
              }
            }
            mRefO(qh_id, d, b) = acc * inv_sum;
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

    auto [Q, PageSize, D, H, PageCount, B] = problem_shape;
    auto [D_latent, D_rope] = D;

    int H_R = H;

    typename MlaKernel::CollectiveMainloop::Fusion::Arguments fusion;

    mutlass::FastDivmod divmod_hr(H_R);
    fusion.fast_divmod_hr = divmod_hr;

    typename MlaKernel::Arguments arguments {
      problem_shape,
      {
        block_Q.get(), stride_Q_latent,
        block_Q.get() + D_latent, stride_Q_rope,
        block_C.get(), stride_C_latent,
        block_C.get() + D_latent, stride_K_rope,
        block_PageTable.get(), stride_PT,
        block_Seq.get(),
        1.0f / sqrtf(128 + D_rope),
        fusion
      },
      {
        block_O.get(), stride_O,
        block_LSE.get(), stride_LSE
      }
    };

    typename MlaKernel::Params params = MlaKernel::to_underlying_arguments(arguments);


    // QH is grouped
    dim3 grid_dim {
                    static_cast<uint32_t>(ceil_div(Q*H, get<0>(TileShape{}))),
                    static_cast<uint32_t>(num_splits),
                    static_cast<uint32_t>(get<5>(problem_shape))
    };

    mutlass::device_kernel<MlaKernel><<<grid_dim, MlaKernel::MaxThreadsPerBlock, MlaKernel::SharedStorageSize>>>(params);

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
      int total_seq = 0;
      for (int i = 0; i < B; ++i) {
        total_seq += data_Seq[i];
      }
      double flops = 2.0 * Q * total_seq * (D_latent + D_rope + D_latent) * H;
      double bytes = (Q * H * (D_latent + D_rope) * B
                   + total_seq * 1 * (D_latent + D_rope)
                   + Q * H * D_latent * B) * sizeof(Element);


      // warmup
      for (int i = 0; i < 1; ++i) {
        mutlass::device_kernel<MlaKernel><<<grid_dim, MlaKernel::MaxThreadsPerBlock>>>(params);
      }

      GPU_Clock timer;
      timer.start();
      for (int i = 0; i < options.loop; ++i) {
        mutlass::device_kernel<MlaKernel><<<grid_dim, MlaKernel::MaxThreadsPerBlock>>>(params);
      }

      double time = timer.seconds() / float(options.loop);
      double gflops = flops / double(1e9);
      double gbytes = bytes / double (1e9);
      double throughput = gflops / time;
      double bandwidth = gbytes / time;


      printf("R+W:%f GB\n", gbytes);

      printf("MLA decode:[%6.1f] GFLOPS, [%6.1f] GB/s, (%6.4f) ms\n", throughput, bandwidth, time * 1000.f);

    }
    return 0;
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

  MlaRunner runner;

  return runner.run(options);
}
