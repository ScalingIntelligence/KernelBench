from __future__ import annotations

from typing import Optional

import torch

from mate import msa_interface as _msa

__all__ = [
    "SparseDecodePagedAttentionWrapper",
    "SparseK2qCsrBuilderSm100",
    "fp4_indexer_block_scores",
    "sparse_atten_nvfp4_kv_func",
    "sparse_decode_atten_func",
    "sparse_fmha",
    "sparse_fmha_plan",
]


def sparse_fmha_plan(*args, **kwargs):
    return _msa.sparse_msa_plan(*args, **kwargs)


def sparse_fmha(*args, **kwargs):
    return _msa.sparse_msa(*args, **kwargs)


def _page_table_to_flat_kv_indices(
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    *,
    page_size: int,
    device: torch.device,
) -> torch.Tensor:
    if page_table.ndim != 2:
        raise ValueError(f"page_table must be rank-2, got {tuple(page_table.shape)}")
    page_table = page_table.to(device=device, dtype=torch.int32, non_blocking=True)
    page_counts = torch.div(
        cache_seqlens + int(page_size) - 1,
        int(page_size),
        rounding_mode="floor",
    )
    if int(page_counts.numel()) != int(page_table.shape[0]):
        raise ValueError(
            "page_table batch mismatch: "
            f"{int(page_table.shape[0])} vs {int(page_counts.numel())}"
        )
    max_required_pages = int(page_counts.max().item()) if page_counts.numel() else 0
    if int(page_table.shape[1]) < max_required_pages:
        raise ValueError(
            "page_table does not contain enough columns for cache_seqlens: "
            f"{int(page_table.shape[1])} < {max_required_pages}"
        )
    slices = [
        page_table[batch_idx, : int(count)]
        for batch_idx, count in enumerate(page_counts.to("cpu").tolist())
        if int(count) > 0
    ]
    if slices:
        return torch.cat(slices, dim=0).contiguous()
    return torch.empty((0,), dtype=torch.int32, device=device)


def _q2k_to_kv_block_indexes(
    q2k_indices: Optional[torch.Tensor],
    *,
    total_q: int,
    num_kv_heads: int,
) -> Optional[torch.Tensor]:
    if q2k_indices is None:
        return None
    if q2k_indices.ndim != 3:
        raise ValueError(
            "q2k_indices must have shape [Hkv, total_q, topK] or "
            f"[total_q, Hkv, topK], got {tuple(q2k_indices.shape)}"
        )
    if int(q2k_indices.shape[0]) == num_kv_heads:
        if int(q2k_indices.shape[1]) != total_q:
            raise ValueError(
                "q2k_indices total_q mismatch: "
                f"{int(q2k_indices.shape[1])} vs {total_q}"
            )
        return q2k_indices.permute(1, 0, 2).contiguous()
    if int(q2k_indices.shape[0]) == total_q:
        if int(q2k_indices.shape[1]) != num_kv_heads:
            raise ValueError(
                "q2k_indices num_kv_heads mismatch: "
                f"{int(q2k_indices.shape[1])} vs {num_kv_heads}"
            )
        return q2k_indices.contiguous()
    raise ValueError(
        "q2k_indices must have shape [Hkv, total_q, topK] or "
        f"[total_q, Hkv, topK], got {tuple(q2k_indices.shape)}"
    )


def sparse_decode_atten_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: Optional[torch.Tensor] = None,
    *,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int,
    max_seqlen_k: int,
    blk_kv: int = 128,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    schedule=None,
    O_partial: Optional[torch.Tensor] = None,
    LSE_partial: Optional[torch.Tensor] = None,
):
    """MSA-style sparse decode wrapper backed by ``mate.msa_interface``."""

    del schedule, O_partial, LSE_partial
    if q.ndim != 3:
        raise ValueError(f"q must have shape [B * Sq, Hq, D], got {tuple(q.shape)}")
    if k.ndim != 4 or v.ndim != 4:
        raise ValueError("k and v must be paged tensors with rank 4")
    if int(k.shape[2]) == int(blk_kv):
        num_kv_heads = int(k.shape[1])
    elif int(k.shape[1]) == int(blk_kv):
        num_kv_heads = int(k.shape[2])
    else:
        raise ValueError(
            "k must use [pages, Hkv, page, D] or [pages, page, Hkv, D], got "
            f"{tuple(k.shape)} with blk_kv={blk_kv}"
        )
    if q.shape[0] % int(seqlen_q) != 0:
        raise ValueError(
            f"q.shape[0] ({q.shape[0]}) must be divisible by seqlen_q ({seqlen_q})"
        )
    batch_size = int(q.shape[0]) // int(seqlen_q)
    if int(page_table.shape[0]) != batch_size:
        raise ValueError(
            f"page_table batch mismatch: {int(page_table.shape[0])} vs {batch_size}"
        )
    qo_lens = torch.full(
        (batch_size,),
        int(seqlen_q),
        dtype=torch.int32,
        device=q.device,
    )
    cache_seqlens = seqused_k.to(device=q.device, dtype=torch.int32, non_blocking=True)
    if int(cache_seqlens.numel()) != batch_size:
        raise ValueError(
            f"seqused_k batch mismatch: {int(cache_seqlens.numel())} vs {batch_size}"
        )
    kv_lens = cache_seqlens
    kv_block_indexes = _q2k_to_kv_block_indexes(
        q2k_indices,
        total_q=int(q.shape[0]),
        num_kv_heads=num_kv_heads,
    )
    topk = int(kv_block_indexes.shape[-1]) if kv_block_indexes is not None else 0
    if kv_block_indexes is not None and topk != 16:
        raise ValueError(f"MSA forward requires topK=16, got {topk}")
    if kv_block_indexes is None:
        kv_indices = _page_table_to_flat_kv_indices(
            page_table,
            cache_seqlens,
            page_size=int(blk_kv),
            device=q.device,
        )
        if return_softmax_lse:
            raise NotImplementedError(
                "fmha_sm100 sparse_decode_atten_func dense all-KV fallback "
                "does not implement return_softmax_lse yet"
            )
        plan = _msa._msa_plan_from_lengths(
            qo_lens,
            kv_lens,
            num_qo_heads=int(q.shape[1]),
            num_kv_heads=num_kv_heads,
            page_size=int(blk_kv),
            num_kv_splits=1,
            causal=bool(causal),
            split_prefill_decode=False,
        )
        out, _ = _msa.msa(
            q,
            k,
            v,
            plan,
            kv_indices=kv_indices,
            sm_scale=softmax_scale,
        )
        return out
    plan = _msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads=int(q.shape[1]),
        num_kv_heads=num_kv_heads,
        page_size=int(blk_kv),
        num_kv_splits=1,
        causal=bool(causal),
        kv_block_num=topk,
        sparse_kernel_mode="decode",
        split_prefill_decode=False,
    )
    result = _msa.sparse_decode_atten_func(
        q,
        k,
        v,
        plan,
        page_table=page_table,
        kv_block_indexes=kv_block_indexes,
        sm_scale=softmax_scale,
        return_softmax_lse=return_softmax_lse,
    )
    if not return_softmax_lse:
        return result
    out, lse = result
    if lse.shape == (q.shape[1], q.shape[0]):
        lse = lse.transpose(0, 1).contiguous()
    return out, lse


class SparseDecodePagedAttentionWrapper:
    """Minimal MSA-compatible plan/run helper backed by ``sparse_decode_atten_func``."""

    def __init__(self, *, blk_kv: int = 128, causal: bool = True):
        self.blk_kv = int(blk_kv)
        self.causal = bool(causal)
        self.page_table: Optional[torch.Tensor] = None
        self.seqused_k: Optional[torch.Tensor] = None
        self.q2k_indices: Optional[torch.Tensor] = None
        self.seqlen_q: Optional[int] = None
        self.max_seqlen_k: Optional[int] = None

    def plan(
        self,
        *,
        page_table: torch.Tensor,
        seqused_k: torch.Tensor,
        seqlen_q: int,
        max_seqlen_k: int,
        q2k_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> "SparseDecodePagedAttentionWrapper":
        unsupported = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "enable_cuda_graph",
                "fixed_split_size",
                "disable_split_kv",
                "max_grid_size",
            }
            and value not in (None, False)
        }
        if unsupported:
            raise NotImplementedError(
                "SparseDecodePagedAttentionWrapper does not implement "
                f"{sorted(unsupported)} yet"
            )
        self.page_table = page_table
        self.seqused_k = seqused_k
        self.q2k_indices = q2k_indices
        self.seqlen_q = int(seqlen_q)
        self.max_seqlen_k = int(max_seqlen_k)
        return self

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        softmax_scale: Optional[float] = None,
        return_softmax_lse: bool = False,
    ):
        if (
            self.page_table is None
            or self.seqused_k is None
            or self.seqlen_q is None
            or self.max_seqlen_k is None
        ):
            raise RuntimeError(
                "SparseDecodePagedAttentionWrapper.plan() must run first"
            )
        return sparse_decode_atten_func(
            q,
            k,
            v,
            self.q2k_indices,
            page_table=self.page_table,
            seqused_k=self.seqused_k,
            seqlen_q=self.seqlen_q,
            max_seqlen_k=self.max_seqlen_k,
            blk_kv=self.blk_kv,
            causal=self.causal,
            softmax_scale=softmax_scale,
            return_softmax_lse=return_softmax_lse,
        )


def sparse_atten_nvfp4_kv_func(*args, **kwargs):
    raise NotImplementedError("fmha_sm100 MSA wrapper does not implement NVFP4 KV yet")


def fp4_indexer_block_scores(*args, **kwargs):
    raise NotImplementedError(
        "fmha_sm100 MSA wrapper does not implement FP4 indexer yet"
    )


class SparseK2qCsrBuilderSm100:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "SparseK2qCsrBuilderSm100 was removed with the legacy k-centric "
            "prefill path; MSA forward consumes q2k block indices directly"
        )
