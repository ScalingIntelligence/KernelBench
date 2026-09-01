# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch_musa  # noqa: F401
import math
import mate
from mate.jit.runtime import ffi_to_torch  # noqa: F401
from mate.mha_interface import flash_attn_with_kvcache
from mate.testing import supported_musa_compute_capability


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    # v = ckv.repeat_interleave(num_heads, dim=1)
    v = ckv.view(batch_size, kv_len, 1, ckv_dim).repeat_interleave(num_heads, dim=2)

    return k, v


def generate_kv_from_cache_varlen(ckv, kpe, batch_size, num_heads_q, num_heads_k):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size * page_num, page_size, num_heads_k, ckv_dim)
    kpe = kpe.view(batch_size * page_num, page_size, num_heads_k, kpe_dim)
    k = torch.cat([ckv, kpe], dim=-1).repeat_interleave(num_heads_q, dim=-2)
    v = ckv.repeat_interleave(num_heads_q // num_heads_k, dim=-2)

    return k, v


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    batch_size = q.shape[0]
    qo_len = q.shape[1]
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[2]
    head_dim_qk = q.shape[3]
    head_dim_vo = v.shape[3]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .to(q)
    )

    # return o_ref, lse_ref * math.log2(math.e)
    return o_ref, lse_ref


def attention_ref_varlen(
    q_cache: torch.Tensor,  # (batch * seqlen, num_heads, head_dim)
    key_cache: torch.Tensor,  # (page_num, page_size, num_heads, head_dim)
    value_cache: torch.Tensor,  # (page_num, page_size, num_heads, head_dim)
    kv_lens,
    cu_seqlens_q,
    page_table,
    causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(kv_lens)
    qo_len = q_cache.shape[0] // batch_size
    num_q_heads = q_cache.shape[-2]
    num_kv_heads = key_cache.shape[-2]
    head_size_qk = key_cache.shape[-1]
    head_size_v = value_cache.shape[-1]
    page_size = key_cache.shape[1]
    o_ref = []
    lse_ref = []

    for b in range(batch_size):
        kv_len = kv_lens[b]
        q = q_cache[b * qo_len : (b + 1) * qo_len]
        if cu_seqlens_q is not None:
            qo_len = cu_seqlens_q[b + 1] - cu_seqlens_q[b]
            q = q_cache[cu_seqlens_q[b] : cu_seqlens_q[b + 1]]

        num_kv_blocks = (kv_len + page_size - 1) // page_size
        block_indices = page_table[b, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size_qk)
        k = k[:kv_len].repeat_interleave(num_q_heads // num_kv_heads, dim=-2)
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size_v)
        v = v[:kv_len].repeat_interleave(num_q_heads // num_kv_heads, dim=-2)
        logits = torch.einsum("qhd,khd->hqk", q, k).float() * sm_scale

        empty_mask = torch.ones(qo_len, kv_len, device=q_cache.device)
        mask = torch.triu(empty_mask, diagonal=kv_len - qo_len + 1).bool()
        if causal:
            logits = logits.masked_fill_(mask, float("-inf"))
        lse = torch.logsumexp(logits, -1)
        lse_ref.append(lse)
        logits = torch.softmax(logits, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", logits, v)
        o_ref.append(out)
    return torch.cat(o_ref, dim=0), torch.cat(lse_ref, dim=1)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch_size", [1, 56, 60])
@pytest.mark.parametrize("kv_len", [33, 97, 129])
@pytest.mark.parametrize("qo_len", [1, 3, 5])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("schedule_mode", ["inited", "uninited"])
def test_fa_interface(
    batch_size, kv_len, qo_len, num_heads, causal, page_size, dtype, schedule_mode
):
    device = torch.device("musa")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        batch_size, qo_len, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size, qo_len, num_heads, head_dim_kpe, dtype=dtype, device=device
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / ((512 + 64) ** 0.5)  # use head dimension before matrix absorption
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    page_table = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    ).reshape(batch_size, pages_num)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)

    # ckv&kpe should be contiguous
    kv_concat = torch.cat([ckv, kpe], dim=-1)
    use_flash_mla_asm = num_heads == 128
    k_cache = kv_concat[:, :, head_dim_ckv:]
    v_cache = kv_concat[:, :, :head_dim_ckv]
    if not use_flash_mla_asm:
        k_cache = k_cache.unsqueeze(2)
        v_cache = v_cache.unsqueeze(2)

    workspace_buffer = torch.empty(128 * 1024 * 1024, device=device, dtype=torch.uint8)

    scheduler_metadata = None

    if use_flash_mla_asm and schedule_mode == "inited":
        # Used to init schedule info
        flash_attn_with_kvcache(
            q_pe,
            k_cache,
            v_cache,
            qv=q_nope,
            cache_seqlens=kv_lens,
            page_table=page_table,
            softmax_scale=sm_scale,
            causal=causal,
            scheduler_metadata=(workspace_buffer, False),
            return_softmax_lse=True,
        )
        scheduler_metadata = (workspace_buffer, True)
    elif use_flash_mla_asm and schedule_mode == "uninited":
        scheduler_metadata = (workspace_buffer, False)

    o, lse, *rest = flash_attn_with_kvcache(
        q_pe,
        k_cache,
        v_cache,
        qv=q_nope,
        cache_seqlens=kv_lens,
        page_table=page_table,
        softmax_scale=sm_scale,
        causal=causal,
        scheduler_metadata=scheduler_metadata,
        return_softmax_lse=True,
    )
    if not use_flash_mla_asm:
        lse = lse.transpose(1, 2).contiguous()

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(o, o_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("batch_size", [1, 56, 60])
@pytest.mark.parametrize("kv_len", [33, 97, 129])
@pytest.mark.parametrize("qo_len", [1, 3, 5])
@pytest.mark.parametrize("num_heads", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
def test_flashmla_interface(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    dtype,
):
    device = torch.device("musa")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        qo_len, num_heads, batch_size, head_dim_ckv, dtype=dtype, device=device
    ).permute(2, 0, 1, 3)
    q_pe = torch.randn(
        qo_len, num_heads, batch_size, head_dim_kpe, dtype=dtype, device=device
    ).permute(2, 0, 1, 3)
    pages_num = math.ceil(kv_len / page_size)
    num_layers = 3
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        num_layers,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        num_layers,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    ckv = ckv[:, :, 1, :]
    kpe = kpe[:, :, 1, :]

    sm_scale = 1.0 / ((512 + 64) ** 0.5)  # use head dimension before matrix absorption
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    page_table = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    ).reshape(batch_size, pages_num)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)
    q = torch.cat([q_nope, q_pe], dim=-1)

    o_ref, lse_ref = attention_ref(
        batch_size, q.to("cpu"), k.to("cpu"), v.to("cpu"), causal, sm_scale
    )
    o_ref = o_ref.to(device)
    lse_ref = lse_ref.to(device)

    tile_scheduler_metadata, num_splits = mate.flashmla.get_mla_metadata(
        kv_lens, qo_len * num_heads // 1, 1
    )
    o, lse = mate.flashmla.flash_mla_with_kvcache(
        q,
        torch.cat([ckv, kpe], dim=-1).unsqueeze(-2),
        page_table,
        kv_lens,
        head_dim_ckv,
        tile_scheduler_metadata,
        num_splits,
        sm_scale,
        causal,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(o, o_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("num_heads_q", [8, 16, 32, 64, 128])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("batch_size", [1, 32, 56])
@pytest.mark.parametrize("kv_len", [1024])
@pytest.mark.parametrize("max_seqlen_q", [3])
@torch.inference_mode()
def test_mla_decode_varlen(
    num_heads_q: tuple[int, int],
    dtype: torch.dtype,
    block_size: int,
    is_causal,
    batch_size,
    kv_len,
    max_seqlen_q,
) -> None:
    device = torch.device("musa")

    num_heads_k = 1

    qo_len = torch.randint(1, max_seqlen_q + 1, (batch_size,))
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    cu_seqlens_q = (
        torch.nn.functional.pad(torch.cumsum(qo_len, dim=0), (1, 0))
        .to(torch.int32)
        .to(device)
    )
    head_dim_ckv = 512
    head_dim_kpe = 64
    q = torch.randn(
        sum(qo_len),
        num_heads_q,
        head_dim_ckv + head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    q_nope = q[:, :, :head_dim_ckv]
    q_pe = q[:, :, head_dim_ckv:]
    pages_num = math.ceil(max(kv_lens) / block_size)
    ckv = torch.randn(
        batch_size * pages_num,
        block_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        block_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / (
        (head_dim_ckv + head_dim_kpe) ** 0.5
    )  # use head dimension before matrix absorption
    kv_lens = kv_lens.to(torch.int32).to(device)

    page_table = torch.arange(
        0,
        batch_size * pages_num,
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, pages_num)
    k, v = generate_kv_from_cache_varlen(ckv, kpe, batch_size, num_heads_q, num_heads_k)

    kv_concat = torch.cat([ckv, kpe], dim=-1).unsqueeze(2)
    o_ref, lse_ref = attention_ref_varlen(
        q, k, v, kv_lens, cu_seqlens_q, page_table, is_causal, sm_scale
    )
    scheduler_metadata = None
    if num_heads_q == 128:
        workspace_buffer = torch.zeros(
            128 * 1024 * 1024, device=device, dtype=torch.uint8
        )
        scheduler_metadata = (workspace_buffer, False)
    o, lse, *rest = flash_attn_with_kvcache(
        q=q_pe,
        k_cache=kv_concat[..., head_dim_ckv:],
        v_cache=kv_concat[..., :head_dim_ckv],
        qv=q_nope,
        cache_seqlens=kv_lens,  # cache_seqlens
        page_table=page_table,  # page_table
        cu_seqlens_q=cu_seqlens_q,  # cu_query_lens
        max_seqlen_q=max_seqlen_q,  # max_seqlen_q
        softmax_scale=sm_scale,
        causal=is_causal,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        rotary_interleaved=False,
        scheduler_metadata=scheduler_metadata,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
        return_softmax_lse=True,
    )
    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(o, o_ref, atol=atol, rtol=rtol)


@supported_musa_compute_capability([31])
@pytest.mark.parametrize("num_heads_q", [32, 128])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("is_varlen", [True, False])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("kv_len", [128])
@pytest.mark.parametrize("max_seqlen_q", [3])
@torch.inference_mode()
def test_mla_q_not_contig(
    num_heads_q: tuple[int, int],
    dtype: torch.dtype,
    block_size: int,
    is_causal,
    is_varlen,
    batch_size,
    kv_len,
    max_seqlen_q,
) -> None:
    device = torch.device("musa")

    head_dim_ckv = 512
    head_dim_kpe = 64
    num_heads_k = 1
    sm_scale = 1.0 / (
        (head_dim_ckv + head_dim_kpe) ** 0.5
    )  # use head dimension before matrix absorption

    qo_len = None
    cu_seqlens_q = None
    if not is_varlen:
        q = torch.randn(
            batch_size,
            num_heads_q,
            max_seqlen_q,
            head_dim_ckv + head_dim_kpe,
            dtype=dtype,
            device=device,
        ).transpose(1, 2)

    else:
        qo_len = torch.randint(1, max_seqlen_q + 1, (batch_size,))
        cu_seqlens_q = (
            torch.nn.functional.pad(torch.cumsum(qo_len, dim=0), (1, 0))
            .to(torch.int32)
            .to(device)
        )

        q = torch.randn(
            num_heads_q,
            sum(qo_len),
            head_dim_ckv + head_dim_kpe,
            dtype=dtype,
            device=device,
        ).transpose(0, 1)

    q_nope = q[:, :, :head_dim_ckv]
    q_pe = q[:, :, head_dim_ckv:]

    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    pages_num = math.ceil(max(kv_lens) / block_size)
    ckv = torch.randn(
        batch_size * pages_num,
        block_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        block_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    page_table = torch.arange(
        0,
        batch_size * pages_num,
        dtype=torch.int32,
        device=device,
    ).reshape(batch_size, pages_num)

    if not is_varlen:
        k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads_q)
        o_ref, lse_ref = attention_ref(batch_size, q, k, v, is_causal, sm_scale)

    else:
        k, v = generate_kv_from_cache_varlen(
            ckv, kpe, batch_size, num_heads_q, num_heads_k
        )
        o_ref, lse_ref = attention_ref_varlen(
            q, k, v, kv_lens, cu_seqlens_q, page_table, is_causal, sm_scale
        )

    kv_concat = torch.cat([ckv, kpe], dim=-1).unsqueeze(2)

    if not is_varlen:
        tile_scheduler_metadata, num_splits = mate.flashmla.get_mla_metadata(
            kv_lens, max_seqlen_q * num_heads_q // 1, 1
        )
        o, lse = mate.flashmla.flash_mla_with_kvcache(
            q,
            kv_concat,
            page_table,
            kv_lens,
            head_dim_ckv,
            tile_scheduler_metadata,
            num_splits,
            sm_scale,
            is_causal,
        )
    else:
        scheduler_metadata = None
        if num_heads_q == 128:
            workspace_buffer = torch.zeros(
                128 * 1024 * 1024, device=device, dtype=torch.uint8
            )
            scheduler_metadata = (workspace_buffer, False)
        o, lse, *rest = flash_attn_with_kvcache(
            q=q_pe,
            k_cache=kv_concat[..., head_dim_ckv:],
            v_cache=kv_concat[..., :head_dim_ckv],
            qv=q_nope,
            cache_seqlens=kv_lens,  # cache_seqlens
            page_table=page_table,  # page_table
            cu_seqlens_q=cu_seqlens_q,  # cu_query_lens
            max_seqlen_q=max_seqlen_q,  # max_seqlen_q
            softmax_scale=sm_scale,
            causal=is_causal,
            window_size=(-1, -1),
            attention_chunk=0,
            softcap=0.0,
            rotary_interleaved=False,
            scheduler_metadata=scheduler_metadata,
            num_splits=1,
            pack_gqa=None,
            sm_margin=0,
            return_softmax_lse=True,
        )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(lse, lse_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(o, o_ref, atol=atol, rtol=rtol)
