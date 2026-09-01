# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import random
import inspect

import pytest
import torch
import torch_musa  # noqa: F401
import pdb  # noqa: F401

from einops import repeat, rearrange
from typing import Optional, Union, Tuple  # noqa: F401

from mate import flash_attn_combine, flash_attn_varlen_func, flash_attn_with_kvcache
from mate.mha_interface import get_scheduler_metadata
from mate.testing.flash_attn import (
    attention_ref,
    attention_accum_ref,  # noqa: F401
    attention_combine_ref,  # noqa: F401
    pad_accum,
    gen_input_tensor,
    gen_seqlen_data,
    mask_unused,
    generate_block_kvcache,
    gen_padding_mask_from_seqlens,
    generate_random_padding_mask,
    apply_rotary_emb,
    lse_ref_from_score,
    unpad_input,
    pad_input,
    _combine_cp_partials,
    make_cp_rank_local_paged_kvcache,
    arange_along_dim,  # noqa: F401
)
from mate.execution_context import (
    maybe_fake_tensor_mode,
    empty_if_dry_run,
    is_fake_mode,
    is_dry_run_enabled,
)
from mate.testing import supported_musa_compute_capability
from torch.testing import assert_close as cmp  # noqa: F401
from mate.jit.attention.fmha.fmha_utils import round_multiple

# from mate.jit.fmha import _fmha_fwd as jit_fmha_fwd  # noqa: F401

USE_FAKE_MODE = is_dry_run_enabled()


def test_flash_attn_combine_signature_matches_fa3() -> None:
    params = inspect.signature(flash_attn_combine).parameters
    assert list(params) == ["out_partial", "lse_partial", "out", "out_dtype"]


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True, False])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("page_size", [1, 16, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 16) for _ in range(1000)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_metadata(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    page_size: int,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
    # total_seqlen_qo = sum(seqlens_qo)
    cu_seqlens_qo = torch.cumsum(
        torch.tensor([0] + seqlens_qo, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    query_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    query_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, None, device
    )
    query_unpad, indices_query, cu_seqlens_query, max_seqlen_query, *rest = unpad_input(
        query_pad, query_padding_mask
    )

    seqlens_kv = [x[1] for x in seq_lens]
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)
    # total_seqlen_kv = sum(seqlens_kv)
    # cu_seqlens_kv = torch.cumsum(
    #     torch.tensor([0] + seqlens_kv, dtype=torch.int32, device=device), dim=0
    # )
    cache_seqlens = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = generate_block_kvcache(
        max_seqlen_kv,
        page_size,
        batch_size,
        head_kv,
        headdim_qk,
        headdim_vo,
        device,
        dtype,
    )

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    learnable_sink = None

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")

    atol, rtol = 1.5e-2, 1e-2

    if num_splits >= 0:
        batch_rounded = round_multiple(batch_size, 4)
        metadata = empty_if_dry_run(
            get_scheduler_metadata,
            empty_values=torch.empty(
                (4 * batch_rounded,), dtype=torch.int32, device=device
            ),
        )(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=cache_seqlens,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=page_size,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse, out_accum, lse_accum, *rest = flash_attn_with_kvcache(
        q=query_unpad,
        k_cache=k_cache if page_size is None else k_cache_paged,
        v_cache=v_cache if page_size is None else v_cache_paged,
        k=None,
        v=None,
        qv=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k_new=None,
        max_seqlen_q=max_seqlen_qo,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_softmax_lse=True,
    )

    if num_splits >= 0:
        out_accum = torch.utils.dlpack.from_dlpack(out_accum)
        lse_accum = torch.utils.dlpack.from_dlpack(lse_accum)
        out_accum_pad, lse_accum_pad = pad_accum(
            out_accum=out_accum,
            lse_accum=lse_accum,
            seqused_q=None,
            cu_seqlens_q=cu_seqlens_qo,
        )
    out_ref, _, score_ref = attention_ref(
        q=query_pad,
        k=k_cache_rep,
        v=v_cache_rep,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)
    lse_ref, _ = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=learnable_sink,
    )

    # batch size of 1000 should never splits
    splits = metadata[:batch_size]
    torch.testing.assert_close(splits, torch.ones_like(splits), atol=0, rtol=0)
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


# ==============================================================================
# ============================== flash_attn_varlen_func ========================
# ==============================================================================


@supported_musa_compute_capability([31])
@pytest.mark.parametrize(
    ("cu_seqlens_list", "max_seqlen"),
    [
        ([0, 128], 128),
        ([0, 64, 128], 64),
    ],
)
@torch.inference_mode()
def test_varlen_mubin_preserves_noncontiguous_k_stride(
    cu_seqlens_list: list[int],
    max_seqlen: int,
):
    torch.manual_seed(123)
    torch.musa.manual_seed(123)

    device = "musa"
    total_seqlen = cu_seqlens_list[-1]
    num_heads = 24
    head_dim = 64
    dtype = torch.float16

    qkv = torch.randn(total_seqlen, 3, num_heads, head_dim, device=device, dtype=dtype)
    q, k_strided, v = qkv.unbind(dim=1)
    assert k_strided.stride() == (3 * num_heads * head_dim, head_dim, 1)
    assert not k_strided.is_contiguous()
    cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int32, device=device)

    out_strided = flash_attn_varlen_func(
        q,
        k_strided,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        backend="mubin",
    )
    out_contiguous = flash_attn_varlen_func(
        q,
        k_strided.contiguous(),
        v,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        backend="mubin",
    )

    torch.testing.assert_close(out_strided, out_contiguous, atol=1e-3, rtol=1e-3)


@supported_musa_compute_capability([31])
@torch.inference_mode()
def test_mubin_accepts_large_singleton_batch_strides():
    torch.manual_seed(123)
    torch.musa.manual_seed(123)

    device = "musa"
    batch = 1
    seqlen = 128
    num_heads = 2
    head_dim = 64
    dtype = torch.float16
    large_batch_stride = 2**31 + 1024

    def make_large_batch_stride_view() -> tuple[torch.Tensor, torch.Tensor]:
        base = torch.randn(seqlen, num_heads, head_dim, device=device, dtype=dtype)
        view = base.as_strided(
            (batch, seqlen, num_heads, head_dim),
            (large_batch_stride, num_heads * head_dim, head_dim, 1),
        )
        regular = base.unsqueeze(0).contiguous()
        return view, regular

    q, q_regular = make_large_batch_stride_view()
    k, k_regular = make_large_batch_stride_view()
    v, v_regular = make_large_batch_stride_view()

    assert q.stride(0) > torch.iinfo(torch.int32).max

    out_strided = flash_attn_varlen_func(q, k, v, backend="mubin")
    out_regular = flash_attn_varlen_func(
        q_regular, k_regular, v_regular, backend="mubin"
    )

    torch.testing.assert_close(out_strided, out_regular, atol=1e-3, rtol=1e-3)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [False])
@pytest.mark.parametrize("mask", [None])
@pytest.mark.parametrize("attention_chunk", [0, 65])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "seq_lens",
    [
        (333, 444),
        (888, 1328),
        (222, 463),
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (96, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128)])
@pytest.mark.parametrize("backend", ["auto", "mutlass"])
@torch.inference_mode()
def test_varlen_func_bshd(
    batch_size: int,
    seq_lens: tuple[int, int],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    attention_chunk: int,
    pack_gqa: bool,
    num_splits: int,
    backend: str,
):
    device = "musa"

    seqlen_qo = seq_lens[0]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    query, _, _ = gen_input_tensor(
        batch_size=batch_size,
        seqlens=seqlen_qo,
        num_head=head_qo,
        headdim=headdim_qk,
        use_cu_seqlens=False,
        datatype=dtype,
        device=device,
        randop=torch.randn,
    )

    seqlen_kv = seq_lens[1]
    head_kv = num_heads[0]
    headdim_vo = headdim[1]
    key, _, _ = gen_input_tensor(
        batch_size=batch_size,
        seqlens=seqlen_kv,
        num_head=head_kv,
        headdim=headdim_qk,
        use_cu_seqlens=False,
        datatype=dtype,
        device=device,
        randop=torch.randn,
    )
    value, _, _ = gen_input_tensor(
        batch_size=batch_size,
        seqlens=seqlen_kv,
        num_head=head_kv,
        headdim=headdim_vo,
        use_cu_seqlens=False,
        datatype=dtype,
        device=device,
        randop=torch.randn,
    )

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    softmax_scale = headdim_qk**-0.5

    metadata = None
    if num_splits >= 0:
        batch_rounded = round_multiple(batch_size, 4)
        metadata = empty_if_dry_run(
            get_scheduler_metadata,
            empty_values=torch.empty(
                (4 * batch_rounded,), dtype=torch.int32, device=device
            ),
        )(
            batch_size=batch_size,
            max_seqlen_q=seqlen_qo,
            max_seqlen_k=seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=None,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )

    out, lse = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
        seqused_q=None,
        seqused_k=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        learnable_sink=None,
        attention_chunk=attention_chunk,
        softcap=0.0,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=False,
        sm_margin=0,
        return_softmax_lse=True,
        backend=backend,
    )

    out_ref, _, score_ref = attention_ref(
        q=query,
        k=key,
        v=value,
        query_padding_mask=None,
        key_padding_mask=None,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=attention_chunk,
        sink_token_length=0,
        learnable_sink=None,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    _, lse_ref_pad = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        learnable_sink=None,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse_ref_pad, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out, out_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("cp_world_size", [1, 2, 4])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize(
    "seq_lens",  # (seqlen_q, seqlen_kv) or (seqlen_q, seqlen_kv, seqlen_used_q, seqlen_used_kv)
    [
        [(32, 512), (192, 192), (128, 256)],
        [(32, 512, 32, -1), (192, 192, 64, 128), (128, 256, 64, 128)],
        [(i, i + 1234) for i in range(11)],
        [(55, 666), (256, 463), (111, 1328)],
        [(111, 1328), (55, 666), (222, 463)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (96, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@pytest.mark.parametrize("backend", ["auto", "mutlass"])
@torch.inference_mode()
def test_varlen_func_ragged_qkv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
    cp_world_size: int,
    backend: str,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    # prepare seqlen
    seqlens_qo, seqlens_kv, _, _, seqused_qo, seqused_kv = gen_seqlen_data(
        seq_lens, device
    )

    # prepare q data
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)

    q_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    q_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, seqused_qo, device
    )
    q_unpad, indices_q, cu_seqlens_qo, _, *rest = unpad_input(q_pad, q_padding_mask)

    # prepare kv data
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)

    k_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_qk, dtype=dtype, device=device
    )
    v_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_vo, dtype=dtype, device=device
    )
    kv_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_kv, batch_size, seqused_kv, device
    )
    k_unpad, indices_k, cu_seqlens_kv, _, *rest = unpad_input(k_pad, kv_padding_mask)
    v_unpad, indices_v, cu_seqlens_kv, _, *rest = unpad_input(v_pad, kv_padding_mask)

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    learnable_sink = None
    atol, rtol = 1.5e-2, 1e-2

    if cp_world_size > 1:
        # CP path: split KV interleaved across ranks, combine partial outputs.
        # Use effective seqlens (seqused if present) as the split boundary.
        if is_fake_mode():
            eff_seqlens_kv = [0 for _ in range(batch_size)]
        else:
            eff_seqlens_kv = [
                seqused_kv[b].item() if seqused_kv is not None else seqlens_kv[b]
                for b in range(batch_size)
            ]
        cp_tot_seqused_k = torch.tensor(
            eff_seqlens_kv, dtype=torch.int32, device=device
        )

        rank_outs, rank_lses = [], []
        for cp_rank in range(cp_world_size):
            k_segs, v_segs, local_seqlens_k = [], [], []
            for b in range(batch_size):
                sk = eff_seqlens_kv[b]
                k_segs.append(k_pad[b, cp_rank:sk:cp_world_size].contiguous())
                v_segs.append(v_pad[b, cp_rank:sk:cp_world_size].contiguous())
                local_seqlens_k.append(
                    (sk - cp_rank + cp_world_size - 1) // cp_world_size
                )

            cu_seqlens_k_local = torch.cumsum(
                torch.tensor([0] + local_seqlens_k, dtype=torch.int32, device=device),
                dim=0,
                dtype=torch.int32,
            )
            k_local = torch.cat(k_segs, dim=0)
            v_local = torch.cat(v_segs, dim=0)

            metadata = None
            if num_splits >= 0:
                batch_rounded = round_multiple(batch_size, 4)
                metadata = empty_if_dry_run(
                    get_scheduler_metadata,
                    empty_values=torch.empty(
                        (4 * batch_rounded,), dtype=torch.int32, device=device
                    ),
                )(
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_qo,
                    max_seqlen_k=max(local_seqlens_k),
                    num_heads_q=head_qo,
                    num_heads_kv=head_kv,
                    headdim=headdim_qk,
                    seqused_q=seqused_qo,
                    seqused_k=None,
                    headdim_v=headdim_vo,
                    qkv_dtype=dtype,
                    cu_seqlens_q=cu_seqlens_qo,
                    cu_seqlens_k=cu_seqlens_k_local,
                    cu_seqlens_k_new=None,
                    cache_leftpad=None,
                    page_size=None,
                    max_seqlen_k_new=0,
                    causal=is_causal,
                    window_size=window_size,
                    attention_chunk=0,
                    has_softcap=False,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    mp_margin=0,
                )

            out_rank, lse_rank = flash_attn_varlen_func(
                q=q_unpad,
                k=k_local,
                v=v_local,
                cu_seqlens_q=cu_seqlens_qo,
                cu_seqlens_k=cu_seqlens_k_local,
                max_seqlen_q=max_seqlen_qo,
                max_seqlen_k=max(local_seqlens_k),
                seqused_q=seqused_qo,
                seqused_k=None,
                softmax_scale=softmax_scale,
                causal=is_causal,
                qv=None,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                window_size=window_size,
                learnable_sink=learnable_sink,
                attention_chunk=0,
                softcap=0.0,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                deterministic=False,
                sm_margin=0,
                return_softmax_lse=True,
                backend=backend,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                cp_tot_seqused_k=cp_tot_seqused_k,
            )
            rank_outs.append(out_rank.float())
            rank_lses.append(lse_rank.float())

        # Combine per batch item (Q is ragged via cu_seqlens_qo)
        total_q = q_unpad.shape[0]
        combined_unpad = torch.zeros(
            total_q, head_qo, headdim_vo, dtype=torch.float32, device=device
        )
        for b in range(batch_size):
            s = cu_seqlens_qo[b].item()
            e = cu_seqlens_qo[b + 1].item()
            if s == e:
                continue
            out_b, _ = _combine_cp_partials(
                [rank_outs[r][s:e] for r in range(cp_world_size)],
                [rank_lses[r][:, s:e] for r in range(cp_world_size)],
            )
            combined_unpad[s:e] = out_b

        out_pad = mask_unused(
            pad_input(combined_unpad.to(dtype), indices_q, batch_size, max_seqlen_qo),
            seqused_qo,
        )
    else:
        metadata = None
        if num_splits >= 0:
            batch_rounded = round_multiple(batch_size, 4)
            metadata = empty_if_dry_run(
                get_scheduler_metadata,
                empty_values=torch.empty(
                    (4 * batch_rounded,), dtype=torch.int32, device=device
                ),
            )(
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_qo,
                max_seqlen_k=max_seqlen_kv,
                num_heads_q=head_qo,
                num_heads_kv=head_kv,
                headdim=headdim_qk,
                seqused_q=seqused_qo,
                seqused_k=seqused_kv,
                headdim_v=headdim_vo,
                qkv_dtype=dtype,
                cu_seqlens_q=cu_seqlens_qo,
                cu_seqlens_k=cu_seqlens_kv,
                cu_seqlens_k_new=None,
                cache_leftpad=None,
                page_size=None,
                max_seqlen_k_new=0,
                causal=is_causal,
                window_size=window_size,
                attention_chunk=0,
                has_softcap=False,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                mp_margin=0,
            )

        out, lse = flash_attn_varlen_func(
            q=q_unpad,
            k=k_unpad,
            v=v_unpad,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            seqused_q=seqused_qo,
            seqused_k=seqused_kv,
            softmax_scale=softmax_scale,
            causal=is_causal,
            qv=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            window_size=window_size,
            learnable_sink=learnable_sink,
            attention_chunk=0,
            softcap=0.0,
            scheduler_metadata=metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            deterministic=False,
            sm_margin=0,
            return_softmax_lse=True,
            backend=backend,
        )
        out_pad = mask_unused(
            pad_input(out, indices_q, batch_size, max_seqlen_qo), seqused_qo
        )

    out_ref, _, score_ref = attention_ref(
        q=q_pad,
        k=k_pad,
        v=v_pad,
        query_padding_mask=q_padding_mask,
        key_padding_mask=kv_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    # lse_ref_unpad, _ = lse_ref_from_score(
    #     score_ref,
    #     is_causal=is_causal,
    #     cu_seqlens_q=cu_seqlens_qo,
    #     cu_seqlens_k=cu_seqlens_kv,
    #     seqused_q=seqused_qo,
    #     seqused_k=seqused_kv,
    #     learnable_sink=learnable_sink,
    # )

    out_ref = mask_unused(out_ref, seqused_qo)

    atol, rtol = 1.5e-2, 1e-2

    # pdb.set_trace()

    # torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [False, True])
@pytest.mark.parametrize("is_learnable_sink", [False, True])
@pytest.mark.parametrize("mask", [None, "causal", (44, 88)])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(111, 1328), (55, 666), (222, 463)],
        [(111, 1328, 66, 888), (55, 666, 55, -1), (222, 463, -1, 256)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_varlen_func_ragged_qkv_advance(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    is_learnable_sink: bool,
    num_splits: int,
    softcap: float,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    # prepare seqlen
    seqlens_qo, seqlens_kv, _, _, seqused_qo, seqused_kv = gen_seqlen_data(
        seq_lens, device
    )

    # prepare q data
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)

    q_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    q_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, seqused_qo, device
    )
    q_unpad, indices_q, cu_seqlens_qo, _, *rest = unpad_input(q_pad, q_padding_mask)

    # prepare kv data
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)

    k_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_qk, dtype=dtype, device=device
    )
    v_pad = torch.randn(
        batch_size, max_seqlen_kv, head_kv, headdim_vo, dtype=dtype, device=device
    )
    kv_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_kv, batch_size, seqused_kv, device
    )
    k_unpad, indices_k, cu_seqlens_kv, _, *rest = unpad_input(k_pad, kv_padding_mask)
    v_unpad, indices_v, cu_seqlens_kv, _, *rest = unpad_input(v_pad, kv_padding_mask)

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    is_local = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore
        is_local = True

    if is_learnable_sink:
        if not is_local:
            pytest.skip("learnable sink is tested for local attention")
        learnable_sink = torch.randn(head_qo, device=device, dtype=dtype) * 10
    else:
        learnable_sink = None

    if num_splits >= 0:
        batch_rounded = round_multiple(batch_size, 4)
        metadata = empty_if_dry_run(
            get_scheduler_metadata,
            empty_values=torch.empty(
                (4 * batch_rounded,), dtype=torch.int32, device=device
            ),
        )(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=seqused_qo,
            seqused_k=seqused_kv,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k=cu_seqlens_kv,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=softcap != 0.0,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse = flash_attn_varlen_func(
        q=q_unpad,
        k=k_unpad,
        v=v_unpad,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=cu_seqlens_kv,
        max_seqlen_q=max_seqlen_qo,
        max_seqlen_k=max_seqlen_kv,
        seqused_q=seqused_qo,
        seqused_k=seqused_kv,
        softmax_scale=softmax_scale,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=softcap,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=False,
        sm_margin=0,
        return_softmax_lse=True,
        backend="mutlass",
    )

    out_ref, _, score_ref = attention_ref(
        q=q_pad,
        k=k_pad,
        v=v_pad,
        query_padding_mask=q_padding_mask,
        key_padding_mask=kv_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=softcap,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    # import pdb
    # pdb.set_trace()

    # lse_ref_unpad, lse_ref_pad = lse_ref_from_score(
    #     score_ref,
    #     is_causal=is_causal,
    #     cu_seqlens_q=cu_seqlens_qo,
    #     cu_seqlens_k=cu_seqlens_kv,
    #     seqused_q=seqused_qo,
    #     seqused_k=seqused_kv,
    #     learnable_sink=learnable_sink,
    # )

    out_pad = mask_unused(
        pad_input(out, indices_q, batch_size, max_seqlen_qo), seqused_qo
    )
    out_ref = mask_unused(out_ref, seqused_qo)

    atol, rtol = 1.5e-2, 1e-2

    # torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


# ==============================================================================
# ============================= flash_attn_with_kvcache ========================
# ==============================================================================


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("cp_world_size", [1, 2, 4])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True, False])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("page_size", [1, 3, 4, 16, 64, 111])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(32, 512), (192, 192), (128, 256)],
        [(i, i + 1234) for i in range(11)],
        [(55, 666), (222, 463), (111, 1328)],
        [(111, 1328), (55, 666), (222, 463)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (96, 8), (32, 8), (64, 4), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_paged_attn(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    page_size: int,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
    cp_world_size: int,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
    # total_seqlen_qo = sum(seqlens_qo)
    cu_seqlens_qo = torch.cumsum(
        torch.tensor([0] + seqlens_qo, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    query_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    query_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, None, device
    )
    query_unpad, indices_query, cu_seqlens_query, max_seqlen_query, *rest = unpad_input(
        query_pad, query_padding_mask
    )

    seqlens_kv = [x[1] for x in seq_lens]
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)
    # total_seqlen_kv = sum(seqlens_kv)
    # cu_seqlens_kv = torch.cumsum(
    #     torch.tensor([0] + seqlens_kv, dtype=torch.int32, device=device), dim=0
    # )
    cache_seqlens = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = generate_block_kvcache(
        max_seqlen_kv,
        page_size,
        batch_size,
        head_kv,
        headdim_qk,
        headdim_vo,
        device,
        dtype,
    )

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore

    learnable_sink = None

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")

    atol, rtol = 1.5e-2, 1e-2

    if cp_world_size > 1:
        cp_tot_seqused_k = cache_seqlens.clone()

        rank_outs, rank_lses = [], []
        for cp_rank in range(cp_world_size):
            # pre-gather rank-local paged KV
            k_local, v_local, pt_local, local_seqlens = (
                make_cp_rank_local_paged_kvcache(
                    k_cache_paged,
                    v_cache_paged,
                    page_table,
                    cache_seqlens,
                    page_size,
                    cp_rank,
                    cp_world_size,
                    device,
                )
            )

            metadata = None
            if num_splits >= 0:
                batch_rounded = round_multiple(batch_size, 4)
                metadata = empty_if_dry_run(
                    get_scheduler_metadata,
                    empty_values=torch.empty(
                        (4 * batch_rounded,), dtype=torch.int32, device=device
                    ),
                )(
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_qo,
                    max_seqlen_k=1 if is_fake_mode() else max(local_seqlens).item(),
                    num_heads_q=head_qo,
                    num_heads_kv=head_kv,
                    headdim=headdim_qk,
                    seqused_q=None,
                    seqused_k=local_seqlens,
                    headdim_v=headdim_vo,
                    qkv_dtype=dtype,
                    cu_seqlens_q=cu_seqlens_qo,
                    cu_seqlens_k=None,
                    cu_seqlens_k_new=None,
                    cache_leftpad=None,
                    page_size=page_size,
                    max_seqlen_k_new=0,
                    causal=is_causal,
                    window_size=window_size,
                    attention_chunk=0,
                    has_softcap=False,
                    num_splits=num_splits,
                    pack_gqa=pack_gqa,
                    mp_margin=0,
                )

            out_rank, lse_rank, *_ = flash_attn_with_kvcache(
                q=query_unpad,
                k_cache=k_local,
                v_cache=v_local,
                cache_seqlens=local_seqlens,  # rank-local seqlen
                page_table=pt_local,
                cu_seqlens_q=cu_seqlens_qo,
                max_seqlen_q=max_seqlen_qo,
                softmax_scale=softmax_scale,
                causal=is_causal,
                window_size=window_size,
                learnable_sink=learnable_sink,
                scheduler_metadata=metadata,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=0,
                return_softmax_lse=True,
                cp_world_size=cp_world_size,
                cp_rank=cp_rank,
                cp_tot_seqused_k=cp_tot_seqused_k,
            )
            rank_outs.append(out_rank.float())
            rank_lses.append(lse_rank.float())

        total_q = query_unpad.shape[0]
        combined_unpad = torch.zeros(
            total_q, head_qo, headdim_vo, dtype=torch.float32, device=device
        )
        cu_q = cu_seqlens_qo.tolist()
        for b in range(batch_size):
            s, e = cu_q[b], cu_q[b + 1]
            if s == e:
                continue
            out_b, _ = _combine_cp_partials(
                [rank_outs[r][s:e] for r in range(cp_world_size)],
                [rank_lses[r][:, s:e] for r in range(cp_world_size)],
            )
            combined_unpad[s:e] = out_b

        out_pad = pad_input(
            combined_unpad.to(dtype), indices_query, batch_size, max_seqlen_query
        )
    else:
        metadata = None
        if num_splits >= 0:
            batch_rounded = round_multiple(batch_size, 4)
            metadata = empty_if_dry_run(
                get_scheduler_metadata,
                empty_values=torch.empty(
                    (4 * batch_rounded,), dtype=torch.int32, device=device
                ),
            )(
                batch_size=batch_size,
                max_seqlen_q=max_seqlen_qo,
                max_seqlen_k=max_seqlen_kv,
                num_heads_q=head_qo,
                num_heads_kv=head_kv,
                headdim=headdim_qk,
                seqused_q=None,
                seqused_k=cache_seqlens,
                headdim_v=headdim_vo,
                qkv_dtype=dtype,
                cu_seqlens_q=cu_seqlens_qo,
                cu_seqlens_k=None,
                cu_seqlens_k_new=None,
                cache_leftpad=None,
                page_size=page_size,
                max_seqlen_k_new=0,
                causal=is_causal,
                window_size=window_size,
                attention_chunk=0,
                has_softcap=False,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                mp_margin=0,
            )
            # print(metadata[:4*batch_size].view(4, -1))

            # TODO: too many feat unsupported
            # out_accum_ref, lse_accum_ref = attention_accum_ref(
            #     query=query_pad,
            #     key=k_cache_rep,
            #     value=v_cache_rep,
            #     metadata=metadata,
            #     num_splits=num_splits,
            #     softmax_scale=softmax_scale,
            #     seqused_q=None,
            #     seqused_kv=cache_seqlens,
            #     cu_seqlens_q=cu_seqlens_qo,
            #     cu_seqlens_kv=None,
            # )

        out, lse, out_accum, lse_accum, *rest = flash_attn_with_kvcache(
            q=query_unpad,
            k_cache=k_cache if page_size is None else k_cache_paged,
            v_cache=v_cache if page_size is None else v_cache_paged,
            k=None,
            v=None,
            qv=None,
            rotary_cos=None,
            rotary_sin=None,
            cache_seqlens=cache_seqlens,
            cache_batch_idx=None,
            cache_leftpad=None,
            page_table=page_table,
            cu_seqlens_q=cu_seqlens_qo,
            cu_seqlens_k_new=None,
            max_seqlen_q=max_seqlen_qo,
            rotary_seqlens=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            softmax_scale=softmax_scale,
            causal=is_causal,
            window_size=window_size,
            learnable_sink=learnable_sink,
            attention_chunk=0,
            softcap=0.0,
            rotary_interleaved=False,
            scheduler_metadata=metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=0,
            return_softmax_lse=True,
        )

        if num_splits >= 0:
            out_accum = torch.utils.dlpack.from_dlpack(out_accum)
            lse_accum = torch.utils.dlpack.from_dlpack(lse_accum)
            out_accum_pad, lse_accum_pad = pad_accum(
                out_accum=out_accum,
                lse_accum=lse_accum,
                seqused_q=None,
                cu_seqlens_q=cu_seqlens_qo,
            )
        out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)

    out_ref, _, score_ref = attention_ref(
        q=query_pad,
        k=k_cache_rep,
        v=v_cache_rep,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )
    lse_ref, _ = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=learnable_sink,
    )

    atol, rtol = 1.5e-2, 1e-2
    if cp_world_size == 1:
        torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True, False])
@pytest.mark.parametrize("is_learnable_sink", [True, False])
@pytest.mark.parametrize("is_varlen_q", [True, False])
@pytest.mark.parametrize("mask", [None, "causal", (44, 88)])
@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(111, 1328), (55, 666), (222, 463)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(40, 8), (8, 1)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_paged_attn_advance(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    page_size: int,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    is_learnable_sink: bool,
    is_varlen_q: bool,
    num_splits: int,
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
    # total_seqlen_qo = sum(seqlens_qo)
    cu_seqlens_qo = torch.cumsum(
        torch.tensor([0] + seqlens_qo, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )
    query_pad = torch.randn(
        batch_size, max_seqlen_qo, head_qo, headdim_qk, device=device, dtype=dtype
    )
    query_padding_mask = gen_padding_mask_from_seqlens(
        seqlens_qo, batch_size, None, device
    )
    query_unpad, indices_query, cu_seqlens_query, max_seqlen_query, *rest = unpad_input(
        query_pad, query_padding_mask
    )

    seqlens_kv = [x[1] for x in seq_lens]
    head_kv = num_heads[1]
    headdim_vo = headdim[1]
    max_seqlen_kv = max(seqlens_kv)
    # total_seqlen_kv = sum(seqlens_kv)
    # cu_seqlens_kv = torch.cumsum(
    #     torch.tensor([0] + seqlens_kv, dtype=torch.int32, device=device), dim=0
    # )
    cache_seqlens = torch.tensor(seqlens_kv, dtype=torch.int32, device=device)
    (
        k_cache,
        v_cache,
        page_table,
        k_cache_paged,
        v_cache_paged,
        num_blocks,
    ) = generate_block_kvcache(
        max_seqlen_kv,
        page_size,
        batch_size,
        head_kv,
        headdim_qk,
        headdim_vo,
        device,
        dtype,
    )

    softmax_scale = headdim_qk**-0.5

    is_causal = False
    is_local = False
    window_size = (None, None)
    if isinstance(mask, str) and mask == "causal":
        is_causal = True
    elif isinstance(mask, tuple) and len(mask) == 2:
        window_size = mask  # type: ignore
        is_local = True

    if is_learnable_sink:
        if not is_local:
            pytest.skip("learnable sink is tested for local attention")
        learnable_sink = torch.randn(head_qo, device=device, dtype=dtype) * 10
    else:
        learnable_sink = None

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")

    metadata = None
    if num_splits >= 0:
        batch_rounded = round_multiple(batch_size, 4)
        metadata = empty_if_dry_run(
            get_scheduler_metadata,
            empty_values=torch.empty(
                (4 * batch_rounded,), dtype=torch.int32, device=device
            ),
        )(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=cache_seqlens,
            headdim_v=headdim_vo,
            qkv_dtype=dtype,
            cu_seqlens_q=cu_seqlens_qo if is_varlen_q else None,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=page_size,
            max_seqlen_k_new=0,
            causal=is_causal,
            window_size=window_size,
            attention_chunk=0,
            has_softcap=False,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mp_margin=0,
        )
        # print(metadata[:4*batch_size].view(4, -1))

        # TODO: too many feat unsupported
        # out_accum_ref, lse_accum_ref = attention_accum_ref(
        #     query=query_pad,
        #     key=k_cache_rep,
        #     value=v_cache_rep,
        #     metadata=metadata,
        #     num_splits=num_splits,
        #     softmax_scale=softmax_scale,
        #     seqused_q=None,
        #     seqused_kv=cache_seqlens,
        #     cu_seqlens_q=cu_seqlens_qo,
        #     cu_seqlens_kv=None,
        # )

    out, lse, out_accum, lse_accum, *rest = flash_attn_with_kvcache(
        q=query_unpad if is_varlen_q else query_pad,
        k_cache=k_cache if page_size is None else k_cache_paged,
        v_cache=v_cache if page_size is None else v_cache_paged,
        k=None,
        v=None,
        qv=None,
        rotary_cos=None,
        rotary_sin=None,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=None,
        cache_leftpad=None,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_qo if is_varlen_q else None,
        cu_seqlens_k_new=None,
        max_seqlen_q=max_seqlen_qo if is_varlen_q else None,
        rotary_seqlens=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        softmax_scale=softmax_scale,
        causal=is_causal,
        window_size=window_size,
        learnable_sink=learnable_sink,
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_softmax_lse=True,
    )

    # if num_splits >= 0:
    #     out_accum = torch.utils.dlpack.from_dlpack(out_accum)
    #     lse_accum = torch.utils.dlpack.from_dlpack(lse_accum)
    #     if is_varlen_q:
    #         out_accum_pad, lse_accum_pad = pad_accum(
    #             out_accum=out_accum,
    #             lse_accum=lse_accum,
    #             seqused_q=None,
    #             cu_seqlens_q=cu_seqlens_qo,
    #         )
    #     else:
    #         out_accum_pad = out_accum
    #         lse_accum_pad = lse_accum

    out_ref, _, score_ref = attention_ref(
        q=query_pad,
        k=k_cache_rep,
        v=v_cache_rep,
        query_padding_mask=query_padding_mask if is_varlen_q else None,
        key_padding_mask=key_padding_mask,
        key_leftpad=None,
        attn_bias=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=is_causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=window_size,
        attention_chunk=0,
        sink_token_length=0,
        learnable_sink=learnable_sink,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )
    if is_varlen_q:
        out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)
    else:
        out_pad = out
    lse_ref_unpad, lse_ref_pad = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo if is_varlen_q else None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=learnable_sink,
    )
    lse_ref = lse_ref_unpad if is_varlen_q else lse_ref_pad

    atol, rtol = 1.5e-2, 1e-2

    torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("seqlen_q", [1, 3])
@pytest.mark.parametrize("headdim_v", [128, 256])
@pytest.mark.parametrize("num_head", [1, 4, 5, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mode", ["normal", "ragged", "padded"])
@pytest.mark.parametrize("reorder", [False])
@pytest.mark.parametrize("batch_size", [1, 5, 32])
@pytest.mark.parametrize("max_splits", [5])
def test_combine(
    max_splits: int,
    batch_size: int,
    reorder: bool,
    mode: str,
    dtype: torch.dtype,
    num_head: int,
    headdim_v: int,
    seqlen_q: int,
) -> None:
    device = "musa"

    # torch.manual_seed(666)
    # torch.musa.manual_seed(666)

    assert not reorder, "Reorder is not supported yet."

    # Prepare per-batch valid split counts. The FA3 fwd-style combine
    # interface infers the maximum split count from out_partial.shape[0].
    num_splits = torch.randint(
        2, max_splits + 1, (batch_size,), device=device, dtype=torch.int32
    )

    # Prepare data
    cu_seqlens_q, seqused_q = None, None
    if mode in ("padded", "normal"):
        shape_oaccum = (max_splits, batch_size, seqlen_q, num_head, headdim_v)  # type: ignore[assignment]
        shape_lseaccum = (max_splits, batch_size, seqlen_q, num_head)  # type: ignore[assignment]
    elif mode == "ragged":
        if is_fake_mode():
            total_q = batch_size * seqlen_q
        else:
            seqlens_q = torch.randint(
                1, seqlen_q + 1, (batch_size,), device=device, dtype=torch.int32
            )
            cu_seqlens_q = torch.cat(
                [
                    torch.tensor([0], dtype=torch.int32, device=device),
                    seqlens_q.cumsum(dim=0),
                ]
            ).to(torch.int32)
            total_q = cu_seqlens_q[-1].item()
        shape_oaccum = (max_splits, 1, total_q, num_head, headdim_v)  # type: ignore[assignment]
        shape_lseaccum = (max_splits, 1, total_q, num_head)  # type: ignore[assignment]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    lse_storage_shape = shape_lseaccum[:2] + (
        shape_lseaccum[3],
        shape_lseaccum[2],
    )
    if is_fake_mode():
        out_partial = torch.empty(shape_oaccum, device=device, dtype=torch.float32)
        lse_partial = torch.empty(
            lse_storage_shape,
            device=device,
            dtype=torch.float32,
        ).transpose(2, 3)
        flash_attn_combine(out_partial, lse_partial, out_dtype=dtype)
        return

    if mode == "padded":
        seqused_q = torch.randint(
            1, seqlen_q + 1, (batch_size,), device=device, dtype=torch.int32
        )
        # max_seqlen_q = seqlen_q

    out_accum = torch.randn(shape_oaccum, device=device, dtype=torch.float32)
    lse_partial = torch.randn(
        lse_storage_shape,
        device=device,
        dtype=torch.float32,
    ).transpose(2, 3)

    # Mask Seqlen
    if mode == "padded":
        for batch_idx, cur_seqlen_q in enumerate(seqused_q):
            out_accum[:, batch_idx, cur_seqlen_q:, :, :] = 0
            lse_partial[:, batch_idx, cur_seqlen_q:, :] = float("-inf")

    # Mask Splits
    for batch_idx, cur_splits in enumerate(num_splits):
        cur_splits = cur_splits.item()
        if cur_splits < max_splits:
            if mode == "ragged":
                # pdb.set_trace()
                out_accum[
                    cur_splits:,
                    0,
                    cu_seqlens_q[batch_idx] : cu_seqlens_q[batch_idx + 1],
                    :,
                    :,
                ] = 0
                lse_partial[
                    cur_splits:,
                    0,
                    cu_seqlens_q[batch_idx] : cu_seqlens_q[batch_idx + 1],
                    :,
                ] = float("-inf")
            else:
                out_accum[cur_splits:, batch_idx, :, :, :] = 0
                lse_partial[cur_splits:, batch_idx, :, :] = float("-inf")

    out_partial = out_accum.contiguous()
    out, lse = flash_attn_combine(out_partial, lse_partial, out_dtype=dtype)
    if mode == "ragged":
        out = out.squeeze(0)
        lse = lse.squeeze(0)

    ref_out, ref_lse = attention_combine_ref(
        out_accum.transpose(2, 3),
        lse_partial.transpose(2, 3),
    )
    ref_lse = ref_lse.transpose(1, 2)

    if mode == "ragged":
        ref_out.squeeze_(0)
        ref_lse.squeeze_(0)
    elif mode == "padded":
        for batch_idx, cur_seqlen_q in enumerate(seqused_q):
            cur_seqlen_q = cur_seqlen_q.item()
            ref_out[batch_idx, cur_seqlen_q:] = 0
            ref_lse[batch_idx, cur_seqlen_q:] = 0
            out[batch_idx, cur_seqlen_q:] = 0
            lse[batch_idx, cur_seqlen_q:] = 0

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out, ref_out.to(out.dtype), atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("num_splits", [129, 256])
@pytest.mark.parametrize("reuse_outputs", [False, True])
def test_combine_high_splits(
    num_splits: int,
    reuse_outputs: bool,
) -> None:
    device = "musa"
    batch_size, num_head, seqlen_q, headdim_v = 2, 2, 2, 64
    dtype = torch.bfloat16

    out_partial = torch.randn(
        (num_splits, batch_size, seqlen_q, num_head, headdim_v),
        device=device,
        dtype=torch.float32,
    )
    lse_partial = torch.randn(
        (num_splits, batch_size, num_head, seqlen_q),
        device=device,
        dtype=torch.float32,
    ).transpose(2, 3)
    out_buf = (
        torch.empty(
            (batch_size, seqlen_q, num_head, headdim_v),
            device=device,
            dtype=dtype,
        )
        if reuse_outputs
        else None
    )
    out, lse = flash_attn_combine(
        out_partial,
        lse_partial,
        out=out_buf,
        out_dtype=dtype,
    )
    if reuse_outputs:
        assert out is out_buf

    ref_out, ref_lse = attention_combine_ref(
        out_partial.transpose(2, 3),
        lse_partial.transpose(2, 3),
    )
    ref_lse = ref_lse.transpose(1, 2)

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse, ref_lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out, ref_out.to(out.dtype), atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@maybe_fake_tensor_mode(fake=USE_FAKE_MODE)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_splits", [0])
@pytest.mark.parametrize("new_kv", [True], ids=lambda x: "new_kv" if x else "no_new_kv")
@pytest.mark.parametrize("local", [False])
@pytest.mark.parametrize("causal", [True], ids=lambda x: "causal" if x else "noncausal")
@pytest.mark.parametrize("seqlen_new_eq_seqlen_q", [False])
@pytest.mark.parametrize(
    "has_rotary_seqlens",
    [
        True,
    ],
    ids=lambda x: "rotary_seqlens" if x else "no_rotary_seqlens",
)
@pytest.mark.parametrize(
    "rotary_interleaved",
    [
        True,
    ],
    ids=lambda x: "rotary_interleaved" if x else "rotary_contiguous",
)
@pytest.mark.parametrize("rotary_fraction", [0.5])
@pytest.mark.parametrize("page_size", [None, 32, 64])
@pytest.mark.parametrize(
    "has_leftpad", [True], ids=lambda x: "leftpad" if x else "no_leftpad"
)
@pytest.mark.parametrize(
    "has_batch_idx", [True], ids=lambda x: "batch_idx" if x else "no_batch_idx"
)
@pytest.mark.parametrize(
    "pack_gqa", [True], ids=lambda x: "pack_gqa" if x else "no_pack_gqa"
)
@pytest.mark.parametrize(
    "varlen_q", [True], ids=lambda x: "varlen" if x else "nonvarlen"
)
@pytest.mark.parametrize("heads", [(96, 8)], ids=["h96_8"])
@pytest.mark.parametrize(
    "headdim",
    [
        (192, 128),
        (128, 128),
        (64, 256),
        (64, 512),
    ],
    ids=[
        "d192_128",
        "d128_128",
        "d64_256",
        "d64_512",
    ],
)
@pytest.mark.parametrize(
    "has_qv,only_qv",
    [(False, False), (True, False), (True, True)],
    ids=["no_qv", "qv", "only_qv"],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        # (1, 128),
        (1, 339),
        (3, 1024),
        (64, 800),
        # (64, 256),
        (3, 799),
        # (64, 2048),
        (16, 20000),
        # (1, 128 * 64),
        # (16, 128 * 64),
        # (128, 128),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 43, 77],
)
@pytest.mark.parametrize("softcap", [50.0])
@pytest.mark.parametrize("attention_chunk", [65])
@torch.inference_mode()
def test_advance_features(
    batch_size,
    seqlen_q,
    seqlen_k,
    heads,
    headdim,
    has_qv,
    only_qv,
    varlen_q,
    pack_gqa,
    has_batch_idx,
    has_leftpad,
    page_size,
    rotary_fraction,
    rotary_interleaved,
    has_rotary_seqlens,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    num_splits,
    dtype,
    attention_chunk,
    softcap,
) -> None:
    torch.musa.empty_cache()
    _run_test_advance_features(
        batch_size,
        seqlen_q,
        seqlen_k,
        heads,
        headdim,
        varlen_q,
        pack_gqa,
        has_batch_idx,
        has_leftpad,
        page_size,
        rotary_fraction,
        rotary_interleaved,
        has_rotary_seqlens,
        seqlen_new_eq_seqlen_q,
        causal,
        local,
        new_kv,
        num_splits,
        has_qv,
        only_qv,
        dtype,
        attention_chunk,
        softcap,
    )


def _run_test_advance_features(
    batch_size,
    seqlen_q,
    seqlen_k,
    heads,
    headdim,
    varlen_q,
    pack_gqa,
    has_batch_idx,
    has_leftpad,
    page_size,
    rotary_fraction,
    rotary_interleaved,
    has_rotary_seqlens,
    seqlen_new_eq_seqlen_q,
    causal,
    local,
    new_kv,
    num_splits,
    has_qv,
    only_qv,
    dtype,
    attention_chunk,
    softcap,
) -> None:
    # Skip
    if not has_qv and headdim not in ((192, 128), (128, 128)):
        pytest.skip()
    if seqlen_q > seqlen_k and new_kv:
        pytest.skip()
    if not new_kv:
        if rotary_fraction > 0.0 or seqlen_new_eq_seqlen_q or has_rotary_seqlens:
            pytest.skip()
    if rotary_fraction == 0.0:
        if has_rotary_seqlens or not rotary_interleaved:
            pytest.skip()
    if dtype == torch.float8_e4m3fn and has_qv:
        if headdim not in ((64, 256), (64, 512)):
            pytest.skip("float8 qv only supports headdim (64, 256) and (64, 512)")
        if new_kv or rotary_fraction > 0.0:
            pytest.skip("float8 qv does not support append KV or rotary yet")

    input_dtype = dtype
    ref_dtype = torch.bfloat16
    device = "musa"
    torch.manual_seed(666)
    torch.musa.manual_seed(666)
    random.seed(666)

    head_qo, head_kv = heads
    headdim_qk, headdim_vo = headdim

    rotary_dim = math.floor(int(rotary_fraction * headdim_qk) / 16) * 16
    effective_local = local or attention_chunk > 0

    # init_tensors()
    batch_size_cache = batch_size if not has_batch_idx else batch_size * 2
    q = (
        torch.randn(
            batch_size, seqlen_q, head_qo, headdim_qk, device=device, dtype=ref_dtype
        )
        .to(input_dtype)
        .to(ref_dtype)
    )
    qv_tensor = (
        torch.randn(
            batch_size,
            seqlen_q,
            head_qo,
            headdim_vo,
            device=device,
            dtype=ref_dtype,
        )
        .to(input_dtype)
        .to(ref_dtype)
        if has_qv
        else None
    )
    if varlen_q:
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, device, mode="random"
        )
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, *rest = unpad_input(
            q, query_padding_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
        qv_unpad = (
            rearrange(qv_tensor, "b s ... -> (b s) ...")[indices_q]
            if qv_tensor is not None
            else None
        )
    else:
        query_padding_mask = None
        q_unpad = q
        cu_seqlens_q, max_seqlen_q = None, seqlen_q
        qv_unpad = qv_tensor

    window_size = (None, None) if not local else torch.randint(0, seqlen_k, (2,))

    seqlen_new = seqlen_q if seqlen_new_eq_seqlen_q else random.randint(1, seqlen_q)
    cu_seqlens_k_new = None
    key_new_padding_mask = None
    if new_kv:
        k = (
            torch.randn(
                batch_size,
                seqlen_new,
                head_kv,
                headdim_qk,
                device=device,
                dtype=ref_dtype,
            )
            .to(input_dtype)
            .to(ref_dtype)
        )
        v = (
            torch.randn(
                batch_size,
                seqlen_new,
                head_kv,
                headdim_vo,
                device=device,
                dtype=ref_dtype,
            )
            .to(input_dtype)
            .to(ref_dtype)
        )
        if varlen_q:  # k & v are also varlen
            key_new_padding_mask = generate_random_padding_mask(
                seqlen_new, batch_size, device, mode="random"
            )
            k_unpad, indices_k, cu_seqlens_k_new, *rest = unpad_input(
                k, key_new_padding_mask
            )
            v_unpad, *rest = unpad_input(v, key_new_padding_mask)
        else:
            k_unpad, v_unpad = k, v
    else:
        k, v, k_unpad, v_unpad = None, None, None, None
    if page_size is None:
        k_cache = (
            torch.randn(
                batch_size_cache,
                seqlen_k,
                head_kv,
                headdim_qk,
                device=device,
                dtype=ref_dtype,
            )
            .to(input_dtype)
            .to(ref_dtype)
        )
        # v_cache = torch.arange(seqlen_k, device=device, dtype=ref_dtype).view(
        #     1, -1, 1, 1
        # ).expand(batch_size_cache, seqlen_k, head_kv, headdim_vo).to(input_dtype).to(ref_dtype)
        v_cache = (
            torch.randn(
                batch_size_cache,
                seqlen_k,
                head_kv,
                headdim_vo,
                device=device,
                dtype=ref_dtype,
            )
            .to(input_dtype)
            .to(ref_dtype)
        )
        page_table = None
    else:
        (
            k_cache,
            v_cache,
            page_table,
            k_cache_paged,
            v_cache_paged,
            num_pages,
        ) = generate_block_kvcache(
            seqlen_k,
            page_size,
            batch_size_cache,
            head_kv,
            headdim_qk,
            headdim_vo,
            device,
            dtype,
            ref_dtype,
            torch.randn,
        )
    cache_seqlens = torch.randint(
        0 if new_kv else 1,
        # q RoPE consumes S positions for causal/local/chunked attention.
        (
            (
                seqlen_k
                - (
                    seqlen_q
                    if (causal or effective_local) and rotary_dim > 1
                    else seqlen_new
                )
                + 1
            )
            if new_kv
            else (seqlen_k + 1)
        ),
        (batch_size,),
        dtype=torch.int32,
        device=device,
    )
    if has_leftpad:
        if is_fake_mode():
            cache_leftpad = torch.empty(batch_size, dtype=torch.int32, device=device)
        else:
            cache_leftpad = torch.cat(
                [
                    torch.randint(
                        0,
                        cache_seqlens[i].item(),
                        (1,),
                        dtype=torch.int32,
                        device=device,
                    )
                    if cache_seqlens[i].item() > 0
                    else torch.zeros(1, dtype=torch.int32, device=device)
                    for i in range(batch_size)
                ]
            )
    else:
        cache_leftpad = None
    if has_batch_idx:
        cache_batch_idx = torch.randperm(
            batch_size_cache, dtype=torch.int32, device=device
        )[:batch_size]
    else:
        cache_batch_idx = None

    arange = rearrange(torch.arange(seqlen_k, device=device), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    if not new_kv:
        key_padding_mask = arange < cache_seqlens_expanded
    else:
        k_new_seqlens = (
            key_new_padding_mask.sum(-1, keepdims=True) if varlen_q else seqlen_new
        )
        key_padding_mask = arange < cache_seqlens_expanded + k_new_seqlens
    if has_leftpad:
        key_padding_mask = torch.logical_and(
            key_padding_mask, arange >= cache_leftpad.unsqueeze(-1).expand(-1, seqlen_k)
        )

    rotary_seqlens = cache_seqlens if not has_rotary_seqlens else cache_seqlens // 2
    if rotary_dim > 0:
        angle = (
            torch.rand(
                seqlen_k if page_size is None else num_pages * page_size,
                rotary_dim // 2,
                device=device,
            )
            * 2
            * math.pi
        )
        cos = torch.cos(angle).to(dtype=ref_dtype).to(input_dtype).to(ref_dtype)
        sin = torch.sin(angle).to(dtype=ref_dtype).to(input_dtype).to(ref_dtype)
        if not is_fake_mode():
            if causal or effective_local:
                q_ro = apply_rotary_emb(
                    q,
                    cos,
                    sin,
                    seqlen_offsets=rotary_seqlens,
                    interleaved=rotary_interleaved,
                )
            else:
                q_ro = rearrange(
                    apply_rotary_emb(
                        rearrange(q, "b s h d -> b 1 (s h) d"),
                        cos,
                        sin,
                        seqlen_offsets=rotary_seqlens,
                        interleaved=rotary_interleaved,
                    ),
                    "b 1 (s h) d -> b s h d",
                    s=seqlen_q,
                )
            # q_ro = q
            # k_ro = k
            k_ro = apply_rotary_emb(
                k,
                cos,
                sin,
                seqlen_offsets=rotary_seqlens,
                interleaved=rotary_interleaved,
            )
        else:
            q_ro, k_ro = q, k
    else:
        cos, sin = None, None
        q_ro, k_ro = q, k

    if has_batch_idx:
        k_cache_ref = k_cache[cache_batch_idx.to(dtype=torch.long)].clone()
        v_cache_ref = v_cache[cache_batch_idx.to(dtype=torch.long)].clone()
    else:
        k_cache_ref = k_cache.clone()
        v_cache_ref = v_cache.clone()
    if new_kv:
        update_mask = torch.logical_and(
            cache_seqlens_expanded <= arange,
            arange < cache_seqlens_expanded + k_new_seqlens,
        )
        k_to_update = rearrange(k_ro, "b s ... -> (b s) ...")
        v_to_update = rearrange(v, "b s ... -> (b s) ...")
        if varlen_q:
            k_to_update = k_to_update[indices_k]
            v_to_update = v_to_update[indices_k]
        k_cache_ref[update_mask] = k_to_update
        v_cache_ref[update_mask] = v_to_update
    is_fp8 = input_dtype == torch.float8_e4m3fn
    if is_fp8:
        q_descale, k_descale, v_descale = [
            torch.rand(batch_size, head_kv, device=device, dtype=torch.float32) * 2
            for _ in range(3)
        ]
    else:
        q_descale, k_descale, v_descale = None, None, None
    assert all(x is not None for x in (q_descale, k_descale, v_descale)) == is_fp8

    # generate_ref()
    out_ref, _, score_ref = attention_ref(
        q_ro,
        k_cache_ref,
        v_cache_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        qv=qv_tensor,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        key_leftpad=cache_leftpad,
        softcap=softcap,
        only_qv=only_qv,
    )
    out_pt, _, _ = attention_ref(
        q_ro,
        k_cache_ref,
        v_cache_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        qv=qv_tensor,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        key_leftpad=cache_leftpad,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
        only_qv=only_qv,
    )
    # Handle dtype conversion
    q = q.to(input_dtype)
    q_unpad = q_unpad.to(input_dtype)
    k_cache = k_cache.to(input_dtype)
    v_cache = v_cache.to(input_dtype)
    k_cache_paged = k_cache_paged.to(input_dtype) if page_size is not None else None
    v_cache_paged = v_cache_paged.to(input_dtype) if page_size is not None else None
    k = k.to(input_dtype) if k is not None else None
    v = v.to(input_dtype) if v is not None else None
    k_unpad = k_unpad.to(input_dtype) if k_unpad is not None else None
    v_unpad = v_unpad.to(input_dtype) if v_unpad is not None else None
    qv_unpad = qv_unpad.to(input_dtype) if qv_unpad is not None else None
    cos = cos.to(input_dtype) if cos is not None else None
    sin = sin.to(input_dtype) if sin is not None else None

    # call_kernel()
    scheduler_metadata = None
    if num_splits >= 0:
        batch_rounded = round_multiple(batch_size, 4)
        scheduler_metadata = empty_if_dry_run(
            get_scheduler_metadata,
            empty_values=torch.empty(
                4 * batch_rounded,
                device=device,
                dtype=torch.int32,
            ),
        )(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=seqlen_k,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            seqused_q=None,
            seqused_k=cache_seqlens,
            headdim_v=headdim_vo,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=None,
            cu_seqlens_k_new=cu_seqlens_k_new,
            cache_leftpad=cache_leftpad,
            page_size=page_size,
            max_seqlen_k_new=seqlen_new if new_kv else 0,
            causal=causal,
            window_size=window_size,
            attention_chunk=attention_chunk,
            has_softcap=softcap != 0.0,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            has_qv=has_qv,
            mp_margin=0,
            qkv_dtype=input_dtype,
        )
        # print(scheduler_metadata[: 4 * batch_size].view(4, -1))

    out, lse, *rest = flash_attn_with_kvcache(
        q if not varlen_q else q_unpad,
        k_cache if page_size is None else k_cache_paged,
        v_cache if page_size is None else v_cache_paged,
        k if not new_kv or not varlen_q else k_unpad,
        v if not new_kv or not varlen_q else v_unpad,
        qv=qv_unpad,
        rotary_cos=cos,
        rotary_sin=sin,
        cache_seqlens=cache_seqlens,
        cache_batch_idx=cache_batch_idx,
        cache_leftpad=cache_leftpad,
        page_table=page_table,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k_new,
        max_seqlen_q=max_seqlen_q,
        rotary_seqlens=rotary_seqlens if (not has_qv or has_rotary_seqlens) else None,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        causal=causal,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        return_softmax_lse=True,
        pack_gqa=pack_gqa,
        only_qv=only_qv,
    )
    torch.musa.synchronize()
    if varlen_q:
        out = output_pad_fn(out)

    # compare()
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    if new_kv:
        if page_size is None:
            k_cache_select = (
                k_cache.to(ref_dtype)
                if not has_batch_idx
                else k_cache.to(ref_dtype)[cache_batch_idx]
            )
            v_cache_select = (
                v_cache.to(ref_dtype)
                if not has_batch_idx
                else v_cache.to(ref_dtype)[cache_batch_idx]
            )
        else:
            k_cache_select = rearrange(
                k_cache_paged.to(ref_dtype)[
                    (
                        page_table if not has_batch_idx else page_table[cache_batch_idx]
                    ).flatten()
                ],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
            v_cache_select = rearrange(
                v_cache_paged.to(ref_dtype)[
                    (
                        page_table if not has_batch_idx else page_table[cache_batch_idx]
                    ).flatten()
                ],
                "(b nblocks) block_size ... -> b (nblocks block_size) ...",
                b=batch_size,
            )[:, :seqlen_k]
        k_cache_ref = k_cache_ref.to(input_dtype).to(ref_dtype)
        v_cache_ref = v_cache_ref.to(input_dtype).to(ref_dtype)
        # Check Appended V Cache
        try:
            if input_dtype is not torch.float8_e4m3fn:
                cmp(
                    v_cache_select,
                    v_cache_ref,
                    rtol=0,
                    atol=0,
                )
            else:
                # For FP8, we allow larger tolerance due to the limited precision.
                cmp(
                    v_cache_select,
                    v_cache_ref,
                    rtol=1e-3,
                    atol=1e-3,
                )
        except:
            # pdb.set_trace()
            raise
        # Check Appended K Cache
        if rotary_dim == 0:
            at, rt = 0.0, 0.0
        else:
            if input_dtype is not torch.float8_e4m3fn:
                at, rt = 1e-2, 1.5e-2
            else:
                at, rt = 1e-1, 1e-1

        try:
            cmp(
                k_cache_select,
                k_cache_ref,
                rtol=rt,
                atol=at,
            )
        except:
            # pdb.set_trace()
            raise
    # mult = 2
    # pdb.set_trace()
    # Check Attention Output
    _, lse_ref = lse_ref_from_score(
        score_ref,
        is_causal=causal,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=None,
    )
    try:
        mult = 4 if input_dtype == torch.float8_e4m3fn else 2
        mult_mean = 3 if input_dtype == torch.float8_e4m3fn else 1.5
        # cmp(out, out_ref, rtol=1e-2, atol=1.5e-2)
        diff_max_ours = (out - out_ref).abs().max().item()
        diff_max_pt = (out_pt - out_ref).abs().max().item()
        diff_mean_ours = (out - out_ref).abs().mean().item()
        diff_mean_pt = (out_pt - out_ref).abs().mean().item()
        assert diff_max_ours <= mult * diff_max_pt + 1e-5
        assert diff_mean_ours <= mult_mean * diff_mean_pt + 1e-5
        # cmp(lse, lse_ref, rtol=1e-2, atol=1.5e-2)
    except:
        # pdb.set_trace()
        raise
