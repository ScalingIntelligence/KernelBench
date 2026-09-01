from typing import Optional, Tuple, Union

import torch

from mate.api_logging import mate_api
from mate.gdn_kernels.tilelang.gdn_kkt_solve import kkt_solve
from mate.gdn_kernels.tilelang.gdn_l2norm import gdn_l2norm_
from mate.gdn_kernels.tilelang.gdn_prefill import fused_chunk_gdn_prefill


@mate_api
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    chunk_size: int = 64,
    is_log_space: bool = True,
    output: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Chunked Gated Delta Rule (GDN) attention for prefill.

    This implements the gated delta rule linear attention mechanism for efficient
    training and inference. Supports both GQA (grouped query attention) and GVA
    (grouped value attention) configurations.

    Args:
        q (torch.Tensor):
            Queries of shape ``[total_seq_len, num_q_heads, head_size]``.
            Must be contiguous and on MUSA.
        k (torch.Tensor):
            Keys of shape ``[total_seq_len, num_k_heads, head_size]``.
            Must be contiguous and on MUSA.
        v (torch.Tensor):
            Values of shape ``[total_seq_len, num_v_heads, head_size]``.
            Must be contiguous and on MUSA.
        g (Optional[torch.Tensor]):
            Multiplicative forget gate (alpha) of shape
            ``[total_seq_len, num_sab_heads]`` where
            ``num_sab_heads = max(num_q_heads, num_v_heads)``. Must be float32.
            If ``is_log_space=True``, values are interpreted as ``log(alpha)``.
            If ``is_log_space=False``, values are interpreted as alpha in
            ``(0, 1]`` and the TileLang kernel converts alpha to log space.
            If None, defaults to all ones. Default: ``None``.
        beta (Optional[torch.Tensor]):
            Update gate (beta) of shape ``[total_seq_len, num_sab_heads]``.
            Must be float32. If None, defaults to all ones. Default: ``None``.
        scale (Optional[float]):
            Scale factor for the attention scores.
            If not provided, defaults to ``1 / sqrt(head_size)``. Default: ``None``.
        initial_state (Optional[torch.Tensor]):
            Initial KV state of shape ``[num_seqs, num_sab_heads, head_size, head_size]``.
            Must be float32. If None, starts from zero state. Default: ``None``.
        output_final_state (bool):
            Whether to output the final state. Default: ``False``.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape ``[num_seqs + 1]``, int64.
            Required for variable-length sequences (varlen mode).
        use_qk_l2norm_in_kernel (bool):
            Whether to L2-normalize Q and K in-place with the TileLang
            normalization kernel before KKT/prefill. Default: ``False``.
        is_log_space (bool):
            Whether ``g`` is already in log space. Default: ``True``.
        output (Optional[torch.Tensor]):
            Pre-allocated output tensor of shape ``[total_seq_len, num_o_heads, head_size]``
            where ``num_o_heads = max(num_q_heads, num_v_heads)``.
            If None, will be allocated automatically. Default: ``None``.
        output_state (Optional[torch.Tensor]):
            Pre-allocated output state tensor of shape
            ``[num_seqs, num_sab_heads, head_size, head_size]``, float32.
            Required if ``output_final_state=True``. Default: ``None``.

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If ``output_final_state=False``: Returns output tensor of shape
              ``[total_seq_len, num_o_heads, head_size]``.
            - If ``output_final_state=True``: Returns tuple of (output, final_state) where
              final_state has shape ``[num_seqs, num_sab_heads, head_size, head_size]``.

    Note:
        - Supports:
          - GQA: ``num_q_heads % num_k_heads == 0`` and ``num_v_heads == num_k_heads``
          - GVA: ``num_q_heads == num_k_heads`` and
            ``num_v_heads % num_q_heads == 0``
        - The final state is in k-last layout ``[N, H, V, K]``.
    """

    assert chunk_size == 64, "current implementation only support chunk_size==64"
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16 or float16."
    )
    squeeze_varlen_output = False

    if cu_seqlens is not None:
        if q.ndim == 3:
            if k.ndim != 3 or v.ndim != 3:
                raise ValueError(
                    "q, k, and v must all be 3D tensors with shape [S, N, D] "
                    "when using `cu_seqlens`."
                )
            if g is not None and g.ndim != 2:
                raise ValueError(
                    "g must be a 2D tensor with shape [S, N] when using `cu_seqlens`."
                )
            if beta is not None and beta.ndim != 2:
                raise ValueError(
                    "beta must be a 2D tensor with shape [S, N] when using `cu_seqlens`."
                )
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)
            if g is not None:
                g = g.unsqueeze(0)
            if beta is not None:
                beta = beta.unsqueeze(0)
            if output is not None:
                if output.ndim != 3:
                    raise ValueError(
                        "output must be a 3D tensor with shape [S, N, D] "
                        "when using unbatched varlen inputs."
                    )
                output = output.unsqueeze(0)
            squeeze_varlen_output = True
        elif q.ndim == 4:
            if q.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                    f"Please flatten variable-length inputs before processing."
                )
        else:
            raise ValueError(
                "q must be a 3D tensor with shape [S, N, D] when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    assert v.shape[2] % k.shape[2] == 0, (
        "num_qk_heads must be divisible to num_v_heads."
    )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        gdn_l2norm_(q)
        gdn_l2norm_(k)

    A = kkt_solve(
        k=k,
        b=beta,
        cu_seqlens=cu_seqlens,
    )

    o, _, final_state = fused_chunk_gdn_prefill(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        output=output,
        output_state=output_state,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=False,
        output_o=True,
        cu_seqlens=cu_seqlens,
        is_log_space=is_log_space,
    )
    o = o.to(q.dtype)
    if squeeze_varlen_output:
        o = o.squeeze(0)
    return o, final_state
