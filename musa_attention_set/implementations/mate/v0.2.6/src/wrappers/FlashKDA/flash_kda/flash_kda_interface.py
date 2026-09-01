from __future__ import annotations

import math
from typing import Optional

import torch

from mate.kda import chunk_kda


def get_workspace_size(T_total: int, H: int, N: int = 1) -> int:
    """Return the compatibility workspace size in bytes."""

    del T_total, H, N
    return 0


def fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> None:
    """FlashKDA forward wrapper backed by ``mate.kda.chunk_kda``."""

    if scale is None:
        scale = 1.0 / math.sqrt(float(q.shape[-1]))

    chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=final_state is not None,
        cu_seqlens=cu_seqlens,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=float(lower_bound),
        use_qk_l2norm_in_kernel=True,
        output=out,
        final_state=final_state,
    )
