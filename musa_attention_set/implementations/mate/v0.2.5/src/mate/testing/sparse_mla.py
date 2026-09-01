"""Reference implementations used by sparse MLA tests.

These helpers intentionally use PyTorch tensor ops and must not be imported from
FlashMLA wrapper hot paths.  Keeping them out of TileLang kernel modules keeps
production files focused on kernels and host dispatch only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def ref_sparse_mla_fwd_interface(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    attn_sink: Optional[torch.Tensor] = None,
    d_v: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MODEL1 sparse prefill reference with optional extra-KV and attn sink."""

    q = q.float()
    kv = kv.float()
    sq, h, dim_q = q.shape
    sk, g, _ = kv.shape
    assert d_v == dim_q
    assert g == 1, "Only MQA (kv_group == 1) is validated for MODEL1 prefill"
    sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale

    pos_orig = torch.arange(indices.shape[-1], device=indices.device).view(1, 1, -1)
    indices_orig = indices.clone()
    if topk_length is not None:
        tl = topk_length.view(-1, 1, 1)
        indices_orig = torch.where(
            pos_orig < tl, indices_orig, torch.full_like(indices_orig, -1)
        )
    invalid_mask_orig = (indices_orig < 0) | (indices_orig >= sk)
    indices_orig_safe = indices_orig.clone()
    indices_orig_safe[invalid_mask_orig] = 0
    gathered_kv_orig = kv.index_select(
        dim=0, index=indices_orig_safe.flatten()
    ).reshape(sq, -1, dim_q)

    if extra_kv is not None:
        assert extra_indices is not None
        extra_kv = extra_kv.float()
        sk_extra = extra_kv.shape[0]
        indices_extra = extra_indices.clone()
        if extra_topk_length is not None:
            pos = torch.arange(indices_extra.shape[-1], device=indices_extra.device)
            pos = pos.view(1, 1, -1)
            etl = extra_topk_length.view(-1, 1, 1)
            indices_extra = torch.where(
                pos < etl, indices_extra, torch.full_like(indices_extra, -1)
            )
        invalid_mask_extra = (indices_extra < 0) | (indices_extra >= sk_extra)
        indices_extra_safe = indices_extra.clone()
        indices_extra_safe[invalid_mask_extra] = 0
        gathered_kv_extra = extra_kv.index_select(
            dim=0, index=indices_extra_safe.flatten()
        ).reshape(sq, -1, dim_q)

        gathered_kv = torch.cat([gathered_kv_orig, gathered_kv_extra], dim=1)
        invalid_mask = torch.cat([invalid_mask_orig, invalid_mask_extra], dim=2)
    else:
        gathered_kv = gathered_kv_orig
        invalid_mask = invalid_mask_orig

    logits = q @ gathered_kv.transpose(1, 2)
    logits *= sm_scale
    logits = logits.masked_fill(invalid_mask.view(sq, 1, -1), float("-inf"))

    lonely_q_mask = invalid_mask.view(sq, -1).all(dim=1).view(sq, 1).expand(sq, h)
    safe_lse = torch.full(
        (sq, h), float("+inf"), dtype=logits.dtype, device=logits.device
    )
    valid_q_mask = ~lonely_q_mask
    safe_lse[valid_q_mask] = torch.logsumexp(logits[valid_q_mask], dim=-1)

    attn = torch.zeros_like(logits)
    attn[valid_q_mask] = torch.exp(
        logits[valid_q_mask] - safe_lse[valid_q_mask].unsqueeze(-1)
    )
    out = attn @ gathered_kv[..., :d_v]
    if attn_sink is not None:
        lse_for_o = torch.logsumexp(
            torch.stack([safe_lse, attn_sink.view(1, h).expand(sq, h)], dim=0),
            dim=0,
        )
        sink_scale = torch.nan_to_num(torch.exp(safe_lse - lse_for_o), nan=0.0)
        out = out * sink_scale.unsqueeze(-1)
    out[lonely_q_mask.unsqueeze(-1).expand_as(out)] = 0.0
    return out.to(torch.bfloat16), safe_lse.to(torch.float32)
