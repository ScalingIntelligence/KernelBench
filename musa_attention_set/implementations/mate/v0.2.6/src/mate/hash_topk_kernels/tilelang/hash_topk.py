import functools
from typing import Literal

import torch

TidDType = Literal["int32", "int64"]

__all__ = ["run_hash_topk"]


def _hash_topk_pass_configs(tilelang):
    return {
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
        tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
    }


def _tensor_dtype_name(tensor: torch.Tensor) -> TidDType:
    if tensor.dtype == torch.int32:
        return "int32"
    if tensor.dtype == torch.int64:
        return "int64"
    raise ValueError(f"expected int32 or int64 tensor, got {tensor.dtype}.")


def _tilelang_dtype(T, dtype: TidDType):
    return T.int32 if dtype == "int32" else T.int64


@functools.lru_cache(maxsize=None)
def _tilelang_hash_topk_warp_kernel(
    topk: int,
    num_fused_shared_experts: int,
    input_id_dtype: TidDType,
    tid2eid_dtype: TidDType,
):
    import tilelang
    import tilelang.language as T

    output_topk = topk + num_fused_shared_experts
    if output_topk > 32:
        raise ValueError("warp hash_topk kernel requires topk + shared <= 32.")

    input_id_ty = _tilelang_dtype(T, input_id_dtype)
    tid2eid_ty = _tilelang_dtype(T, tid2eid_dtype)
    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    vocab_size = T.dynamic("vocab_size")
    router_logits_stride_m = T.dynamic("router_logits_stride_m")
    router_logits_stride_n = T.dynamic("router_logits_stride_n")
    input_ids_stride_m = T.dynamic("input_ids_stride_m")
    tid2eid_stride_m = T.dynamic("tid2eid_stride_m")
    tid2eid_stride_n = T.dynamic("tid2eid_stride_n")

    @tilelang.jit(pass_configs=_hash_topk_pass_configs(tilelang))
    def hash_topk_warp_kernel(
        router_logits: T.StridedTensor[
            (num_tokens, num_experts),
            (router_logits_stride_m, router_logits_stride_n),
            T.float32,
        ],
        input_ids: T.StridedTensor[(num_tokens,), (input_ids_stride_m,), input_id_ty],
        tid2eid: T.StridedTensor[
            (vocab_size, topk),
            (tid2eid_stride_m, tid2eid_stride_n),
            tid2eid_ty,
        ],
        routed_scores: T.Tensor[(num_tokens, output_topk), T.float32],
        routed_ids: T.Tensor[(num_tokens, output_topk), T.int64],
        shared_weight: T.float32,
    ) -> None:
        with T.Kernel(num_tokens, threads=32) as token_id:
            tx = T.get_thread_binding()
            input_id = input_ids[token_id]
            expert_id = T.alloc_local((1,), dtype=tid2eid_ty)
            score = T.alloc_local((1,), dtype=T.float32)
            expert_id[0] = 0
            score[0] = 0.0

            if tx < topk:
                expert_id[0] = tid2eid[input_id, tx]
                logit = router_logits[token_id, expert_id[0]]
                score[0] = T.sqrt(T.log(1.0 + T.exp(logit)))

            denominator = T.warp_reduce_sum(score[0])
            if tx < output_topk:
                if tx < topk:
                    routed_ids[token_id, tx] = T.cast(expert_id[0], T.int64)
                    routed_scores[token_id, tx] = score[0] / T.max(denominator, 1e-20)
                else:
                    routed_ids[token_id, tx] = T.cast(num_experts + tx - topk, T.int64)
                    routed_scores[token_id, tx] = shared_weight

    return hash_topk_warp_kernel


@functools.lru_cache(maxsize=None)
def _tilelang_hash_topk_shared_kernel(
    topk: int,
    num_fused_shared_experts: int,
    input_id_dtype: TidDType,
    tid2eid_dtype: TidDType,
    threads: int = 128,
):
    import tilelang
    import tilelang.language as T

    input_id_ty = _tilelang_dtype(T, input_id_dtype)
    tid2eid_ty = _tilelang_dtype(T, tid2eid_dtype)
    num_tokens = T.dynamic("num_tokens")
    num_experts = T.dynamic("num_experts")
    vocab_size = T.dynamic("vocab_size")
    router_logits_stride_m = T.dynamic("router_logits_stride_m")
    router_logits_stride_n = T.dynamic("router_logits_stride_n")
    input_ids_stride_m = T.dynamic("input_ids_stride_m")
    tid2eid_stride_m = T.dynamic("tid2eid_stride_m")
    tid2eid_stride_n = T.dynamic("tid2eid_stride_n")
    output_topk = topk + num_fused_shared_experts
    reduce_width = 1 << (topk - 1).bit_length()

    @tilelang.jit(pass_configs=_hash_topk_pass_configs(tilelang))
    def hash_topk_shared_kernel(
        router_logits: T.StridedTensor[
            (num_tokens, num_experts),
            (router_logits_stride_m, router_logits_stride_n),
            T.float32,
        ],
        input_ids: T.StridedTensor[(num_tokens,), (input_ids_stride_m,), input_id_ty],
        tid2eid: T.StridedTensor[
            (vocab_size, topk),
            (tid2eid_stride_m, tid2eid_stride_n),
            tid2eid_ty,
        ],
        routed_scores: T.Tensor[(num_tokens, output_topk), T.float32],
        routed_ids: T.Tensor[(num_tokens, output_topk), T.int64],
        shared_weight: T.float32,
    ) -> None:
        with T.Kernel(num_tokens, threads=threads) as token_id:
            tx = T.get_thread_binding()
            scores = T.alloc_shared((topk,), dtype=T.float32)
            expert_ids = T.alloc_shared((topk,), dtype=tid2eid_ty)
            reductions = T.alloc_shared((reduce_width,), dtype=T.float32)
            input_id = input_ids[token_id]

            if tx < reduce_width:
                reductions[tx] = 0.0
            if tx < topk:
                expert_id = tid2eid[input_id, tx]
                expert_ids[tx] = expert_id
                logit = router_logits[token_id, expert_id]
                score = T.sqrt(T.log(1.0 + T.exp(logit)))
                scores[tx] = score
                reductions[tx] = score
            T.sync_threads()

            if reduce_width >= 64:
                if tx < 32:
                    reductions[tx] += reductions[tx + 32]
                T.sync_threads()
            if reduce_width >= 32:
                if tx < 16:
                    reductions[tx] += reductions[tx + 16]
                T.sync_threads()
            if reduce_width >= 16:
                if tx < 8:
                    reductions[tx] += reductions[tx + 8]
                T.sync_threads()
            if reduce_width >= 8:
                if tx < 4:
                    reductions[tx] += reductions[tx + 4]
                T.sync_threads()
            if reduce_width >= 4:
                if tx < 2:
                    reductions[tx] += reductions[tx + 2]
                T.sync_threads()
            if reduce_width >= 2:
                if tx < 1:
                    reductions[tx] += reductions[tx + 1]
                T.sync_threads()

            if tx < output_topk:
                if tx < topk:
                    routed_ids[token_id, tx] = T.cast(expert_ids[tx], T.int64)
                    routed_scores[token_id, tx] = scores[tx] / T.max(
                        reductions[0], 1e-20
                    )
                else:
                    routed_ids[token_id, tx] = T.cast(num_experts + tx - topk, T.int64)
                    routed_scores[token_id, tx] = shared_weight

    return hash_topk_shared_kernel


def run_hash_topk(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
) -> None:
    input_id_dtype = _tensor_dtype_name(input_ids)
    tid2eid_dtype = _tensor_dtype_name(tid2eid)
    topk = tid2eid.shape[1]
    output_topk = topk + num_fused_shared_experts
    if output_topk <= 32:
        kernel = _tilelang_hash_topk_warp_kernel(
            topk,
            num_fused_shared_experts,
            input_id_dtype,
            tid2eid_dtype,
        )
    else:
        threads = 128 if topk <= 8 else 256
        kernel = _tilelang_hash_topk_shared_kernel(
            topk,
            num_fused_shared_experts,
            input_id_dtype,
            tid2eid_dtype,
            threads=threads,
        )

    kernel(
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        1.0 / float(routed_scaling_factor),
    )
