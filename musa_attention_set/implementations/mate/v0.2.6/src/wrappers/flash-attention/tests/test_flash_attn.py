import pytest
import torch

from einops import rearrange, repeat
from typing import Optional, Union, Tuple  # noqa: F401
from flash_attn_interface import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_with_kvcache,
    flash_attn_varlen_func,
    get_scheduler_metadata,
)

from mate.testing.flash_attn import (
    gen_seqlen_data,
    unpad_input,
    gen_padding_mask_from_seqlens,
    generate_block_kvcache,
    attention_ref,
    mask_unused,
    pad_input,
    lse_ref_from_score,
)


@pytest.mark.parametrize("layout", ["stacked", "gqa"])
@torch.inference_mode()
def test_flash_attn_qkvpacked_func_forward(layout):
    torch.manual_seed(666)
    device = "musa"
    dtype = torch.bfloat16

    if layout == "stacked":
        qkv = torch.randn(1, 128, 3, 4, 128, device=device, dtype=dtype)
        q, k, v = qkv.unbind(dim=-3)
        packed_kwargs = {}
    else:
        qkv = torch.randn(1, 128, 8, 128, device=device, dtype=dtype)
        q, k, v = qkv.split([4, 2, 2], dim=-2)
        packed_kwargs = {"num_heads_q": 4}

    out = flash_attn_qkvpacked_func(qkv, causal=True, **packed_kwargs)
    out_ref = flash_attn_func(q, k, v, causal=True)

    torch.testing.assert_close(out, out_ref)


def test_flash_attn_qkvpacked_func_backward():
    torch.manual_seed(666)
    device = "musa"
    dtype = torch.bfloat16

    qkv = torch.randn(1, 128, 3, 4, 128, device=device, dtype=dtype, requires_grad=True)
    q, k, v = [
        tensor.detach().contiguous().requires_grad_() for tensor in qkv.unbind(dim=-3)
    ]
    grad_out = torch.randn(1, 128, 4, 128, device=device, dtype=dtype)

    out = flash_attn_qkvpacked_func(qkv, causal=True)
    out_ref = flash_attn_func(q, k, v, causal=True)
    out.backward(grad_out)
    out_ref.backward(grad_out)
    qkv_grad_ref = torch.stack([q.grad, k.grad, v.grad], dim=-3)

    torch.testing.assert_close(out, out_ref)
    torch.testing.assert_close(qkv.grad, qkv_grad_ref)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [-1, 0, 5])
@pytest.mark.parametrize("pack_gqa", [True])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(32, 512), (192, 192), (128, 256)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(64, 4)],
)
@pytest.mark.parametrize("headdim", [(128, 128), (256, 256)])
@torch.inference_mode()
def test_varlen_func_ragged_qkv(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    headdim: tuple[int, int],
    dtype: torch.dtype,
    mask: Union[None, str, Tuple[int, int]],
    pack_gqa: bool,
    num_splits: int,
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

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            cache_seqlens=seqused_kv,
            headdim_v=headdim_vo,
            cu_seqlens_q=cu_seqlens_qo,
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
            sm_margin=0,
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
        attention_chunk=0,
        softcap=0.0,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        deterministic=False,
        sm_margin=0,
        return_attn_probs=True,
        return_softmax_lse=False,
        sinks=None,
        s_aux=None,
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
        learnable_sink=None,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    lse_ref_unpad, _ = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=cu_seqlens_kv,
        seqused_q=seqused_qo,
        seqused_k=seqused_kv,
        learnable_sink=None,
    )

    out_pad = mask_unused(
        pad_input(out, indices_q, batch_size, max_seqlen_qo), seqused_qo
    )
    out_ref = mask_unused(out_ref, seqused_qo)
    lse_ref = lse_ref_unpad

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("num_splits", [0, 5])  # neg value disable split
@pytest.mark.parametrize("pack_gqa", [True])
@pytest.mark.parametrize("mask", [None, "causal"])
@pytest.mark.parametrize("page_size", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "seq_lens",
    [
        [(32, 512), (192, 192), (128, 256)],
    ],
)
@pytest.mark.parametrize(
    "num_heads",
    [(64, 4)],
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
):
    torch.manual_seed(666)
    torch.musa.manual_seed(666)

    device = "musa"

    batch_size = len(seq_lens)

    seqlens_qo = [x[0] for x in seq_lens]
    head_qo = num_heads[0]
    headdim_qk = headdim[0]
    max_seqlen_qo = max(seqlens_qo)
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

    # calc ref
    k_cache_rep = repeat(k_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    v_cache_rep = repeat(v_cache, "b s h d -> b s (h g) d", g=head_qo // head_kv)
    max_seqlen_kv_arange = rearrange(
        torch.arange(max_seqlen_kv, device=device), "s -> 1 s"
    )
    key_padding_mask = max_seqlen_kv_arange < rearrange(cache_seqlens, "b -> b 1")
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
        learnable_sink=None,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
        intermediate_dtype=None,
    )

    if num_splits >= 0:
        metadata = get_scheduler_metadata(
            batch_size=batch_size,
            max_seqlen_q=max_seqlen_qo,
            max_seqlen_k=max_seqlen_kv,
            num_heads_q=head_qo,
            num_heads_kv=head_kv,
            headdim=headdim_qk,
            cache_seqlens=cache_seqlens,
            qkv_dtype=dtype,
            headdim_v=headdim_vo,
            cu_seqlens_q=cu_seqlens_qo,
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
            sm_margin=0,
        )
    else:
        # not split
        metadata = None  # noqa: F841
        num_splits = -1

    out, lse, _, _, *rest = flash_attn_with_kvcache(
        q=query_unpad,
        k_cache=k_cache_paged,
        v_cache=v_cache_paged,
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
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=0,
        return_softmax_lse=True,
        sinks=None,
        s_aux=None,
    )

    out_pad = pad_input(out, indices_query, batch_size, max_seqlen_query)
    lse_ref, _ = lse_ref_from_score(
        score_ref,
        is_causal=is_causal,
        cu_seqlens_q=cu_seqlens_qo,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=cache_seqlens,
        learnable_sink=None,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse_ref, lse, atol=atol, rtol=rtol)
    torch.testing.assert_close(out_pad, out_ref, atol=atol, rtol=rtol)
