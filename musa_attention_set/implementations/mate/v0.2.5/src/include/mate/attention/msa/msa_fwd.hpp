#pragma once

#include "mate/attention/msa/collective/mp31_msa_fwd_collective_tme_warpspecialized.hpp"
#include "mate/attention/msa/collective/msa_fwd_epilogue.hpp"
#include "mate/attention/msa/collective/msa_fwd_pair_epilogue.hpp"
#include "mate/attention/msa/kernel/msa_fwd_kernel_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_fwd_pair_tile_scheduler.hpp"
#include "mate/attention/msa/kernel/msa_fwd_pair_union_kernel_tme_warpspecialized.hpp"
#include "mate/attention/msa/kernel/msa_fwd_tile_scheduler.hpp"
#include "mate/attention/msa/msa_options.hpp"

namespace mate::attention::msa {

// MSA forward implementation:
//   Element       = FP8 E4M3, FP16, or BF16 for Q/K/V/P/O
//   accumulator   = FP32
//   CTA           = one query x one KV head, logical ratio 8 or 16
//   all supported paths = physical M16 SQMMA tile; ratio-8 tails use TME OOB fill
//   sparse blocks = q2k[total_q, Hkv, 16], block/page size 128
//   output        = direct O + LSE (no split, partial output, or combine)
template <class Element_, int HeadRatio_, class... Options_>
struct MsaFwdBuilder {
  using Element                  = Element_;
  static constexpr int HeadRatio = HeadRatio_;
  static constexpr int QStages   = 1;
  static constexpr int KStages   = 1;
  static constexpr int VStages   = 1;
  static_assert(HeadRatio == 8 || HeadRatio == 16, "MSA forward supports head ratios 8 and 16.");

  using CollectiveMainloop =
      collective::Mp31MsaFwdCollectiveTmeWarpSpecialized<Element, QStages, KStages, VStages, Options_...>;
  using CollectiveEpilogue = collective::MsaFwdEpilogue<Element>;
  using TileScheduler      = MsaFwdTileScheduler;
  using Kernel             = MsaFwdKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
};

// TP8-only union path: two adjacent queries fill one physical M16 tile.  Their
// top-k lists are compacted into a CTA-local union with per-query membership
// masks, so shared blocks reuse one K/V load and one QK/PV sequence.
template <class Element_, class... Options_>
struct MsaFwdPairUnionBuilder {
  using Element                = Element_;
  static constexpr int QStages = 1;
  static constexpr int KStages = 1;
  static constexpr int VStages = 1;

  using CollectiveMainloop = collective::Mp31MsaFwdCollectiveTmeWarpSpecialized<Element,
                                                                                QStages,
                                                                                KStages,
                                                                                VStages,
                                                                                Options_...,
                                                                                PackQueryPair<std::true_type>>;
  using CollectiveEpilogue = collective::MsaFwdPairEpilogue<Element>;
  using TileScheduler      = MsaFwdPairTileScheduler;
  using Kernel = MsaFwdPairUnionKernelTmeWarpSpecialized<CollectiveMainloop, CollectiveEpilogue, TileScheduler>;
};

}  // namespace mate::attention::msa
