#pragma once

#include <sqmma.h>

#include <tl_templates/musa/common/intrin.h>

namespace tl {
namespace sqmma {

struct f16 {};
struct bf16 {};
struct f32 {};
struct tf32 {};
struct s8 {};
struct s32 {};
struct u8 {};
struct e4m3 {};
struct e5m2 {};

template <typename Tag> struct type_traits;

template <> struct type_traits<f16> {
  using element_type = __half;
  using storage_type = __half;
  static constexpr int element_bits = 16;
};

template <> struct type_traits<bf16> {
  using element_type = __mt_bfloat16;
  using storage_type = __mt_bfloat16;
  static constexpr int element_bits = 16;
};

template <> struct type_traits<f32> {
  using element_type = float;
  using storage_type = float;
  static constexpr int element_bits = 32;
};

template <> struct type_traits<tf32> {
  using element_type = mtmusa::sqmma::precision::tf32;
  using storage_type = float;
  static constexpr int element_bits = 32;
};

template <> struct type_traits<s8> {
  using element_type = signed char;
  using storage_type = signed char;
  static constexpr int element_bits = 8;
};

template <> struct type_traits<s32> {
  using element_type = int;
  using storage_type = int;
  static constexpr int element_bits = 32;
};

template <> struct type_traits<u8> {
  using element_type = unsigned char;
  using storage_type = unsigned char;
  static constexpr int element_bits = 8;
};

template <> struct type_traits<e4m3> {
  using element_type = __mt_fp8_e4m3;
  using storage_type = __mt_fp8_e4m3;
  static constexpr int element_bits = 8;
};

template <> struct type_traits<e5m2> {
  using element_type = __mt_fp8_e5m2;
  using storage_type = __mt_fp8_e5m2;
  static constexpr int element_bits = 8;
};

template <typename Tag> struct descriptor_pointer_traits {
  using type = typename type_traits<Tag>::element_type;
};

template <> struct descriptor_pointer_traits<tf32> { using type = float; };

template <typename Tag>
TL_DEVICE const typename descriptor_pointer_traits<Tag>::type *
offset_pointer(const void *base, int element_offset) {
  using DescriptorPointer = typename descriptor_pointer_traits<Tag>::type;
  using Storage = typename type_traits<Tag>::storage_type;
  const auto *storage = reinterpret_cast<const Storage *>(base);
  return reinterpret_cast<const DescriptorPointer *>(storage + element_offset);
}

template <typename Tag>
TL_DEVICE mtmusa::sqmma::swizzle_granularity
get_swizzle_granularity(bool k_major) {
  if (k_major) {
    return mtmusa::sqmma::sg_16_byte;
  }
  constexpr int element_bits = type_traits<Tag>::element_bits;
  return element_bits == 32   ? mtmusa::sqmma::sg_64_byte
         : element_bits == 16 ? mtmusa::sqmma::sg_32_byte
                              : mtmusa::sqmma::sg_16_byte;
}

} // namespace sqmma

template <int M, int N, int K, int NumWarpM, int NumWarpN, int InstM, int InstN,
          int InstK, typename ATag, typename BTag, typename CTag>
TL_DEVICE void sqmma_ss(const void *a, const void *b, void *c,
                        unsigned stride_a, unsigned stride_b, int a_col_major,
                        int b_col_major, bool clear_accum) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ == 310
  namespace mt = mtmusa::sqmma;
  using AElement = typename sqmma::type_traits<ATag>::element_type;
  using BElement = typename sqmma::type_traits<BTag>::element_type;
  using CElement = typename sqmma::type_traits<CTag>::element_type;
  using AStorage = typename sqmma::type_traits<ATag>::storage_type;
  using BStorage = typename sqmma::type_traits<BTag>::storage_type;
  using Accumulator =
      mt::fragment<mt::accumulator, InstM, InstN, InstK, CElement>;

  static_assert(NumWarpM > 0 && NumWarpM % 4 == 0,
                "MP31 SQMMA requires NumWarpM to be a positive multiple of 4");
  static_assert(NumWarpN > 0, "MP31 SQMMA requires NumWarpN > 0");
  static_assert(M % (NumWarpM / 4) == 0,
                "M must be divisible by the M-direction squad count");
  static_assert(N % NumWarpN == 0,
                "N must be divisible by the N-direction squad count");

  constexpr int kSquadThreads = 128;
  constexpr int kSquadM = NumWarpM / 4;
  constexpr int kSquadN = NumWarpN;
  constexpr int kSquadTileM = M / kSquadM;
  constexpr int kSquadTileN = N / kSquadN;
  constexpr int kInstCountM = kSquadTileM / InstM;
  constexpr int kInstCountN = kSquadTileN / InstN;
  constexpr int kInstCountK = K / InstK;
  constexpr int kCElementsPerThread = InstM * InstN / kSquadThreads;
  static_assert(kSquadTileM % InstM == 0 && kSquadTileN % InstN == 0 &&
                    K % InstK == 0,
                "SQMMA native instruction must tile the squad GEMM exactly");
  static_assert(InstM * InstN % kSquadThreads == 0,
                "SQMMA accumulator must distribute evenly over a squad");

  const int squad_id = static_cast<int>(threadIdx.x) / kSquadThreads;
  const int squad_m = squad_id % kSquadM;
  const int squad_n = squad_id / kSquadM;
  const int squad_row = squad_m * kSquadTileM;
  const int squad_col = squad_n * kSquadTileN;

  const auto a_layout = a_col_major ? mt::mem_col_major : mt::mem_row_major;
  const auto b_layout = b_col_major ? mt::mem_col_major : mt::mem_row_major;
  const auto a_swizzle = sqmma::get_swizzle_granularity<ATag>(!a_col_major);
  const auto b_swizzle = sqmma::get_swizzle_granularity<BTag>(b_col_major);
  const unsigned stride_a_bytes = stride_a * sizeof(AStorage);
  const unsigned stride_b_bytes = stride_b * sizeof(BStorage);
  const auto *a_base = reinterpret_cast<const AStorage *>(a);
  const auto *b_base = reinterpret_cast<const BStorage *>(b);
  auto *c_base = reinterpret_cast<CElement *>(c);

#pragma unroll
  for (int inst_n = 0; inst_n < kInstCountN; ++inst_n) {
#pragma unroll
    for (int inst_m = 0; inst_m < kInstCountM; ++inst_m) {
      const int atom_index = inst_n * kInstCountM + inst_m;
      CElement *atom_c = c_base + atom_index * kCElementsPerThread;
      Accumulator accum;
#pragma unroll
      for (int i = 0; i < kCElementsPerThread; ++i) {
        accum.x[i] = atom_c[i];
      }

#pragma unroll
      for (int inst_k = 0; inst_k < kInstCountK; ++inst_k) {
        const int row = squad_row + inst_m * InstM;
        const int col = squad_col + inst_n * InstN;
        const int k = inst_k * InstK;
        const int a_offset =
            a_col_major ? k * stride_a + row : row * stride_a + k;
        const int b_offset =
            b_col_major ? col * stride_b + k : k * stride_b + col;

        mt::sqmmadesc<mt::desc_a, AElement> a_desc;
        mt::sqmmadesc<mt::desc_b, BElement> b_desc;
        mt::make_sqmma_desc(a_desc,
                            sqmma::offset_pointer<ATag>(a_base, a_offset),
                            stride_a_bytes, a_swizzle, mt::ss_256_byte);
        mt::make_sqmma_desc(b_desc,
                            sqmma::offset_pointer<BTag>(b_base, b_offset),
                            stride_b_bytes, b_swizzle, mt::ss_256_byte);
        // MTCC maps init_none to scale_d=0 (clear) and init_zero to
        // scale_d=1 (accumulate the input C fragment).
        const auto scale_out =
            clear_accum && inst_k == 0 ? mt::init_none : mt::init_zero;
        mt::mma_sync(accum, a_desc, b_desc, accum, a_layout, b_layout,
                     mt::positive, mt::positive, scale_out);
      }

#pragma unroll
      for (int i = 0; i < kCElementsPerThread; ++i) {
        atom_c[i] = accum.x[i];
      }
    }
  }
  // MTCC exposes SQMMA issue through mtmusa::sqmma::mma_sync, but the current
  // public header does not provide the matching wait wrapper. Keep the builtin
  // isolated here until MTCC publishes that API.
  __musa_sqmma_wait();
#endif
}

} // namespace tl
