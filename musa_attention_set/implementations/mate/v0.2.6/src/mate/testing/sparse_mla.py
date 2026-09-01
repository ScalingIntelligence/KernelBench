"""Data builders and references for sparse MLA correctness tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch


Variant = Literal["v32", "v4"]


@dataclass(frozen=True)
class SparseMlaCase:
    name: str
    variant: Variant
    q_len: int
    kv_len: int
    heads: int
    topk: int
    batch: int = 1
    page_size: int = 64
    topk_length: bool = False
    attn_sink: bool = False
    invalid_rows: bool = False
    extra_kv_len: int = 0
    extra_topk: int = 0
    full: bool = False

    @property
    def head_dim(self) -> int:
        return 576 if self.variant == "v32" else 512


@dataclass(frozen=True)
class RopeCase:
    name: str
    nnz: int
    heads: int
    is_neox: bool
    cos_sin_dtype: torch.dtype
    full: bool = False


@dataclass(frozen=True)
class Fp8DecodeCase:
    name: str
    batch: int
    q_len: int
    kv_len: int
    topk: int
    page_size: int
    kv_ndim: int
    invalid_rows: bool = False
    full: bool = False


@dataclass
class PrefillData:
    q: torch.Tensor
    kv: torch.Tensor
    indices: torch.Tensor
    topk_length: Optional[torch.Tensor]
    attn_sink: Optional[torch.Tensor]
    ref_out: torch.Tensor
    ref_max_logits: torch.Tensor
    ref_lse: torch.Tensor


@dataclass
class DecodeData:
    q: torch.Tensor
    k_cache: torch.Tensor
    indices: torch.Tensor
    topk_length: Optional[torch.Tensor]
    attn_sink: Optional[torch.Tensor]
    extra_k_cache: Optional[torch.Tensor]
    extra_indices: Optional[torch.Tensor]
    extra_topk_length: Optional[torch.Tensor]
    ref_out: torch.Tensor
    ref_lse: torch.Tensor


@dataclass
class RopeData:
    q_rope: torch.Tensor
    k_rope: torch.Tensor
    q_nope: torch.Tensor
    k_nope: torch.Tensor
    cos_sin_cache: torch.Tensor
    pos_ids: torch.Tensor
    ref_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class Fp8DecodeData:
    query: torch.Tensor
    kv_cache: torch.Tensor
    indices: torch.Tensor
    seq_lens: torch.Tensor
    ref_out: torch.Tensor
    ref_lse: torch.Tensor


def _seed(*values: int) -> None:
    torch.manual_seed(
        20260817 + sum((index + 1) * value for index, value in enumerate(values))
    )


def _make_indices(
    rows: int,
    topk: int,
    kv_len: int,
    device: torch.device | str,
    *,
    invalid_rows: bool,
) -> torch.Tensor:
    indices = torch.full((rows, 1, topk), -1, dtype=torch.int32, device=device)
    valid = min(topk, kv_len)
    if valid:
        token = torch.arange(valid, dtype=torch.int32, device=device).view(1, 1, -1)
        row = torch.arange(rows, dtype=torch.int32, device=device).view(-1, 1, 1)
        indices[..., :valid] = (token + row * 17) % kv_len
        if valid > 7:
            indices[1::2, :, 7] = -1
        if valid > 11:
            indices[2::3, :, 11] = kv_len + 5
    if invalid_rows:
        indices[::3].fill_(-1)
    return indices


def _make_lengths(
    rows: int, topk: int, device: torch.device | str, *, invalid_rows: bool
) -> torch.Tensor:
    lengths = torch.tensor(
        [topk, max(1, topk * 2 // 3), max(1, topk // 3), 0],
        dtype=torch.int32,
        device=device,
    ).repeat((rows + 3) // 4)[:rows]
    if not invalid_rows:
        lengths.clamp_min_(1)
    return lengths.contiguous()


def _make_sink(heads: int, device: torch.device | str) -> torch.Tensor:
    sink = torch.linspace(-1.0, 1.0, heads, dtype=torch.float32, device=device)
    if heads > 1:
        sink[0], sink[-1] = float("-inf"), float("inf")
    return sink


def ref_sparse_mla_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    d_v: int = 512,
    topk_length: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference sparse MQA with FlashMLA invalid-row and sink semantics."""
    q = q.float()
    kv = kv.float()
    rows, heads, head_dim = q.shape

    def select(
        source: torch.Tensor,
        selected: torch.Tensor,
        lengths: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = selected.clone()
        if lengths is not None:
            position = torch.arange(selected.shape[-1], device=selected.device)
            selected.masked_fill_(position.view(1, 1, -1) >= lengths.view(-1, 1, 1), -1)
        invalid = (selected < 0) | (selected >= source.shape[0])
        safe = selected.masked_fill(invalid, 0)
        gathered = source.index_select(0, safe.flatten().long()).reshape(
            rows, -1, head_dim
        )
        return gathered, invalid

    gathered, invalid = select(kv, indices, topk_length)
    if extra_kv is not None:
        assert extra_indices is not None
        extra_gathered, extra_invalid = select(
            extra_kv.float(), extra_indices, extra_topk_length
        )
        gathered = torch.cat((gathered, extra_gathered), dim=1)
        invalid = torch.cat((invalid, extra_invalid), dim=-1)

    logits = (q @ gathered.transpose(1, 2)) * sm_scale
    logits.masked_fill_(invalid.view(rows, 1, -1), float("-inf"))
    all_invalid = invalid.view(rows, -1).all(dim=-1, keepdim=True).expand(rows, heads)
    max_logits = logits.amax(dim=-1)
    finite_lse = torch.logsumexp(logits, dim=-1)
    lse = torch.where(
        all_invalid, torch.full_like(finite_lse, float("inf")), finite_lse
    )
    weights = torch.nan_to_num(torch.softmax(logits, dim=-1), nan=0.0)
    out = weights @ gathered[..., :d_v]
    if attn_sink is not None:
        out *= torch.sigmoid(lse - attn_sink.view(1, heads)).unsqueeze(-1)
    out.masked_fill_(all_invalid.unsqueeze(-1), 0.0)
    return out.to(torch.bfloat16), max_logits.float(), lse.float()


def make_prefill_data(
    case: SparseMlaCase, device: torch.device | str = "musa"
) -> PrefillData:
    _seed(case.q_len, case.kv_len, case.heads, case.topk, case.head_dim)
    q = (
        torch.randn(
            case.q_len, case.heads, case.head_dim, dtype=torch.bfloat16, device=device
        )
        * 0.1
    )
    kv = (
        torch.randn(case.kv_len, 1, case.head_dim, dtype=torch.bfloat16, device=device)
        * 0.1
    )
    indices = _make_indices(
        case.q_len, case.topk, case.kv_len, device, invalid_rows=case.invalid_rows
    )
    topk_length = (
        _make_lengths(case.q_len, case.topk, device, invalid_rows=case.invalid_rows)
        if case.topk_length
        else None
    )
    attn_sink = _make_sink(case.heads, device) if case.attn_sink else None
    ref_out, ref_max_logits, ref_lse = ref_sparse_mla_attention(
        q,
        kv,
        indices,
        sm_scale=case.head_dim**-0.5,
        topk_length=topk_length,
        attn_sink=attn_sink,
    )
    return PrefillData(
        q, kv, indices, topk_length, attn_sink, ref_out, ref_max_logits, ref_lse
    )


def _power_of_two_scale(value: torch.Tensor) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(value, 1e-4).log2().ceil()).float()


def quantize_sparse_mla_cache(kv: torch.Tensor, variant: Variant) -> torch.Tensor:
    """Build the official V3.2 (656-byte) or V4 (584-byte) paged cache."""
    blocks, page_size, kv_heads, head_dim = kv.shape
    assert kv_heads == 1
    kv = kv.squeeze(2)

    if variant == "v32":
        assert head_dim == 576
        cache = torch.empty(
            (blocks, page_size, 656), dtype=torch.uint8, device=kv.device
        )
        nope = cache[..., :512].view(torch.float8_e4m3fn)
        scales = cache[..., 512:528].view(torch.float32)
        rope = cache[..., 528:].view(torch.bfloat16)
        rope.copy_(kv[..., 512:])
        for tile in range(4):
            source = kv[..., tile * 128 : (tile + 1) * 128].float()
            scale = _power_of_two_scale(source.abs().amax(dim=-1) / 448.0)
            scales[..., tile] = scale
            nope[..., tile * 128 : (tile + 1) * 128] = (
                source / scale.unsqueeze(-1)
            ).to(torch.float8_e4m3fn)
        return cache.view(blocks, page_size, 1, 656)

    assert variant == "v4" and head_dim == 512
    bytes_per_token = 584
    padded_page_bytes = ((page_size * bytes_per_token + 575) // 576) * 576
    storage = torch.empty(
        (blocks, padded_page_bytes), dtype=torch.float8_e4m3fn, device=kv.device
    )
    cache = storage[:, : page_size * bytes_per_token]
    nope_rope = cache[:, : page_size * 576].view(blocks, page_size, 576)
    nope = nope_rope[..., :448]
    rope = nope_rope[..., 448:].view(torch.bfloat16)
    scale_bytes = (
        cache[:, page_size * 576 :].view(blocks, page_size, 8).view(torch.uint8)
    )
    rope.copy_(kv[..., 448:])
    scale_bytes[..., 7] = 0
    for tile in range(7):
        source = kv[..., tile * 64 : (tile + 1) * 64].float()
        scale = _power_of_two_scale(source.abs().amax(dim=-1) / 448.0)
        scale_bytes[..., tile] = (scale.log2().to(torch.int32) + 127).to(torch.uint8)
        nope[..., tile * 64 : (tile + 1) * 64] = (source / scale.unsqueeze(-1)).to(
            torch.float8_e4m3fn
        )
    return cache.view(blocks, page_size, 1, bytes_per_token)


def dequantize_sparse_mla_cache(cache: torch.Tensor, variant: Variant) -> torch.Tensor:
    blocks, page_size, kv_heads, bytes_per_token = cache.shape
    assert kv_heads == 1
    result_dim = 576 if variant == "v32" else 512
    result = torch.empty(
        (blocks, page_size, result_dim), dtype=torch.bfloat16, device=cache.device
    )

    if variant == "v32":
        assert bytes_per_token == 656
        cache = cache.view(blocks, page_size, bytes_per_token)
        nope = cache[..., :512].view(torch.float8_e4m3fn)
        scales = cache[..., 512:528].view(torch.float32)
        result[..., 512:] = cache[..., 528:].view(torch.bfloat16)
        for tile in range(4):
            result[..., tile * 128 : (tile + 1) * 128] = nope[
                ..., tile * 128 : (tile + 1) * 128
            ].float() * scales[..., tile].unsqueeze(-1)
    else:
        assert variant == "v4" and bytes_per_token == 584
        flat = cache.view(blocks, -1)
        nope_rope = flat[:, : page_size * 576].view(blocks, page_size, 576)
        nope = nope_rope[..., :448].view(torch.float8_e4m3fn)
        result[..., 448:] = nope_rope[..., 448:].view(torch.bfloat16)
        scale_bytes = (
            flat[:, page_size * 576 :]
            .view(blocks, page_size, 8)[..., :7]
            .view(torch.uint8)
        )
        for tile in range(7):
            scale = (scale_bytes[..., tile].to(torch.int32) << 23).view(torch.float32)
            result[..., tile * 64 : (tile + 1) * 64] = nope[
                ..., tile * 64 : (tile + 1) * 64
            ].float() * scale.unsqueeze(-1)
    return result.view(blocks, page_size, 1, result_dim)


def make_decode_data(
    case: SparseMlaCase, device: torch.device | str = "musa"
) -> DecodeData:
    _seed(
        case.batch,
        case.q_len,
        case.kv_len,
        case.heads,
        case.topk,
        case.page_size,
        case.head_dim,
    )
    pages = (case.kv_len + case.page_size - 1) // case.page_size
    q = (
        torch.randn(
            case.batch,
            case.q_len,
            case.heads,
            case.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    kv = (
        torch.randn(
            pages,
            case.page_size,
            1,
            case.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    )
    indices = _make_indices(
        case.batch * case.q_len,
        case.topk,
        case.kv_len,
        device,
        invalid_rows=case.invalid_rows,
    ).view(case.batch, case.q_len, 1, case.topk)
    topk_length = (
        _make_lengths(case.batch, case.topk, device, invalid_rows=case.invalid_rows)
        if case.topk_length
        else None
    )
    attn_sink = _make_sink(case.heads, device) if case.attn_sink else None
    k_cache = quantize_sparse_mla_cache(kv, case.variant)
    dequantized = dequantize_sparse_mla_cache(k_cache, case.variant).view(
        -1, 1, case.head_dim
    )

    extra_k_cache = extra_indices = extra_topk_length = extra_dequantized = None
    if case.extra_topk:
        extra_pages = (case.extra_kv_len + case.page_size - 1) // case.page_size
        extra_kv = (
            torch.randn(
                extra_pages,
                case.page_size,
                1,
                case.head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            * 0.1
        )
        extra_indices = _make_indices(
            case.batch * case.q_len,
            case.extra_topk,
            case.extra_kv_len,
            device,
            invalid_rows=case.invalid_rows,
        ).view(case.batch, case.q_len, 1, case.extra_topk)
        extra_topk_length = _make_lengths(
            case.batch,
            case.extra_topk,
            device,
            invalid_rows=case.invalid_rows,
        )
        extra_k_cache = quantize_sparse_mla_cache(extra_kv, case.variant)
        extra_dequantized = dequantize_sparse_mla_cache(
            extra_k_cache, case.variant
        ).view(-1, 1, case.head_dim)

    row_lengths = (
        topk_length.repeat_interleave(case.q_len) if topk_length is not None else None
    )
    extra_row_lengths = (
        extra_topk_length.repeat_interleave(case.q_len)
        if extra_topk_length is not None
        else None
    )
    ref_out, _, ref_lse = ref_sparse_mla_attention(
        q.view(-1, case.heads, case.head_dim),
        dequantized,
        indices.view(-1, 1, case.topk),
        sm_scale=case.head_dim**-0.5,
        topk_length=row_lengths,
        attn_sink=attn_sink,
        extra_kv=extra_dequantized,
        extra_indices=(
            extra_indices.view(-1, 1, case.extra_topk)
            if extra_indices is not None
            else None
        ),
        extra_topk_length=extra_row_lengths,
    )
    ref_out = ref_out.view(case.batch, case.q_len, case.heads, 512)
    ref_lse = ref_lse.view(case.batch, case.q_len, case.heads).transpose(1, 2)
    return DecodeData(
        q,
        k_cache,
        indices,
        topk_length,
        attn_sink,
        extra_k_cache,
        extra_indices,
        extra_topk_length,
        ref_out,
        ref_lse,
    )


def _rope_reference(
    value: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    pos_ids: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    cos = cos_sin_cache[pos_ids.long(), :32].float()
    sin = cos_sin_cache[pos_ids.long(), 32:].float()
    while cos.ndim < value.ndim:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    value = value.float()
    left, right = (
        (value[..., :32], value[..., 32:])
        if is_neox
        else (value[..., ::2], value[..., 1::2])
    )
    left_out, right_out = left * cos - right * sin, right * cos + left * sin
    if is_neox:
        return torch.cat((left_out, right_out), dim=-1)
    return torch.stack((left_out, right_out), dim=-1).flatten(-2)


def make_rope_data(case: RopeCase, device: torch.device | str = "musa") -> RopeData:
    _seed(case.nnz, case.heads, int(case.is_neox))
    scale_q, scale_kv = 0.625, 1.75
    q_rope = (
        torch.randn(case.nnz, case.heads, 64, dtype=torch.bfloat16, device=device) * 0.2
    )
    q_nope = (
        torch.randn(case.nnz, case.heads, 512, dtype=torch.bfloat16, device=device)
        * 0.2
    )
    k_rope = torch.randn(case.nnz, 64, dtype=torch.bfloat16, device=device) * 0.2
    k_nope = torch.randn(case.nnz, 512, dtype=torch.bfloat16, device=device) * 0.2
    max_seq_len = max(19, case.nnz * 2)
    angles = torch.randn(max_seq_len, 32, dtype=torch.float32, device=device)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(
        case.cos_sin_dtype
    )
    pos_ids = (
        torch.arange(case.nnz, dtype=torch.int32, device=device) * 7
    ) % max_seq_len
    ref_outputs = (
        (_rope_reference(q_rope, cos_sin_cache, pos_ids, case.is_neox) * scale_q).to(
            torch.float8_e4m3fn
        ),
        (_rope_reference(k_rope, cos_sin_cache, pos_ids, case.is_neox) * scale_kv).to(
            torch.float8_e4m3fn
        ),
        (q_nope.float() * scale_q).to(torch.float8_e4m3fn),
        (k_nope.float() * scale_kv).to(torch.float8_e4m3fn),
    )
    return RopeData(q_rope, k_rope, q_nope, k_nope, cos_sin_cache, pos_ids, ref_outputs)


def make_fp8_decode_data(
    case: Fp8DecodeCase, device: torch.device | str = "musa"
) -> Fp8DecodeData:
    _seed(case.batch, case.q_len, case.kv_len, case.topk, case.page_size)
    heads = 64
    pages = (case.kv_len + case.page_size - 1) // case.page_size
    storage_tokens = pages * case.page_size
    query = (
        torch.randn(
            case.batch,
            case.q_len,
            heads,
            576,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.1
    ).to(torch.float8_e4m3fn)
    flat_kv = (
        torch.randn(storage_tokens, 576, dtype=torch.bfloat16, device=device) * 0.1
    ).to(torch.float8_e4m3fn)
    kv_cache = (
        flat_kv.view(pages, 1, case.page_size, 576)
        if case.kv_ndim == 4
        else flat_kv.view(pages, case.page_size, 576)
    )
    indices = _make_indices(
        case.batch * case.q_len,
        case.topk,
        case.kv_len,
        device,
        invalid_rows=case.invalid_rows,
    ).view(case.batch, case.q_len, case.topk)
    seq_lens = _make_lengths(
        case.batch, case.topk, device, invalid_rows=case.invalid_rows
    )

    ref_out = torch.zeros(
        case.batch,
        case.q_len,
        heads,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    ref_lse = torch.full(
        (case.batch, case.q_len, heads),
        float("inf"),
        dtype=torch.float32,
        device=device,
    )
    flat_kv_float = flat_kv.float()
    for batch_index, active in enumerate(seq_lens.cpu().tolist()):
        for query_index in range(case.q_len):
            selected = indices[batch_index, query_index, :active]
            selected = selected[(selected >= 0) & (selected < storage_tokens)].long()
            if not selected.numel():
                continue
            selected_kv = flat_kv_float.index_select(0, selected)
            scores = query[batch_index, query_index].float() @ selected_kv.T * 0.0625
            ref_out[batch_index, query_index] = (
                torch.softmax(scores, dim=-1) @ selected_kv[:, :512]
            ).to(torch.bfloat16)
            ref_lse[batch_index, query_index] = torch.logsumexp(scores, dim=-1)
    return Fp8DecodeData(query, kv_cache, indices, seq_lens, ref_out, ref_lse)


def make_fp8_decode_stress_data(
    device: torch.device | str = "musa",
) -> Fp8DecodeData:
    """Build the persistent-row workload that exposed the row-barrier race."""
    _seed(128, 1, 64, 2048)
    batch, q_len, heads, topk = 128, 1, 64, 2048
    query = (
        torch.randn(batch, q_len, heads, 576, dtype=torch.bfloat16, device=device) * 0.1
    ).to(torch.float8_e4m3fn)
    kv_cache = (
        torch.randn(1, 1, topk, 576, dtype=torch.bfloat16, device=device) * 0.1
    ).to(torch.float8_e4m3fn)
    indices = torch.full((batch, q_len, topk), -1, dtype=torch.int32, device=device)
    indices[..., 0] = 0
    seq_lens = torch.ones(batch, dtype=torch.int32, device=device)

    selected_kv = kv_cache[0, 0, 0].float()
    ref_out = (
        selected_kv[:512]
        .to(torch.bfloat16)
        .view(1, 1, 1, 512)
        .expand(batch, q_len, heads, 512)
        .contiguous()
    )
    ref_lse = (query.float() * selected_kv.view(1, 1, 1, 576)).sum(dim=-1).mul(0.0625)
    return Fp8DecodeData(query, kv_cache, indices, seq_lens, ref_out, ref_lse)


def assert_sparse_mla_close(
    out: torch.Tensor,
    ref_out: torch.Tensor,
    lse: torch.Tensor,
    ref_lse: torch.Tensor,
    *,
    max_logits: Optional[torch.Tensor] = None,
    ref_max_logits: Optional[torch.Tensor] = None,
) -> None:
    torch.testing.assert_close(out.float(), ref_out.float(), rtol=0.02, atol=0.01)
    torch.testing.assert_close(lse.float(), ref_lse.float(), rtol=0.002, atol=0.02)
    if max_logits is not None:
        assert ref_max_logits is not None
        torch.testing.assert_close(
            max_logits.float(), ref_max_logits.float(), rtol=0.002, atol=0.02
        )
