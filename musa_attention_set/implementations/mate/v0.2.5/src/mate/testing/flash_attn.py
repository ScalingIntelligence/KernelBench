import math

import torch
import torch.nn.functional as F

from einops import rearrange, repeat
from typing import Optional, Tuple
from ..execution_context import is_fake_mode
from ..utils import ceil_div


def _get_seqlen(batch_idx, seqused, cu_seqlens, seqlen):
    if seqused is not None:
        return seqused[batch_idx]

    if cu_seqlens is not None:
        return cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]

    return seqlen


def mask_unused(out, seqused):
    if seqused is None:
        return out

    assert out.shape[0] == seqused.shape[0]

    batch_size = out.shape[0]
    for b in range(batch_size):
        out[b, seqused[b] :, :, :] = 0.0

    return out


def gen_seqlen_data(seqlens, device):
    assert len(seqlens) > 0

    is_used_seqlen = len(seqlens[0]) == 4
    if is_used_seqlen:
        assert all([len(s) == 4 for s in seqlens])
    else:
        assert all([len(s) == 2 for s in seqlens])

    seqused_q = None
    seqused_k = None

    seqlen_q = [s[0] for s in seqlens]
    cu_seqlens_q = torch.tensor(
        [0] + seqlen_q, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)

    seqlen_k = [s[1] for s in seqlens]
    cu_seqlens_k = torch.tensor(
        [0] + seqlen_k, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)

    if is_used_seqlen:
        used_q_ = [s[2] for s in seqlens]
        invalid_used_q = all([s == -1 for s in used_q_])
        if not invalid_used_q:
            used_q = list(
                map(
                    lambda x: min(x[0], x[1]) if x[1] >= 0 else x[0],
                    zip(seqlen_q, used_q_),
                )
            )
            seqused_q = torch.tensor(used_q, dtype=torch.int32, device=device)

        used_k_ = [s[3] for s in seqlens]
        invalid_used_k = all([s == -1 for s in used_k_])
        if not invalid_used_k:
            used_k = list(
                map(
                    lambda x: min(x[0], x[1]) if x[1] >= 0 else x[0],
                    zip(seqlen_k, used_k_),
                )
            )
            seqused_k = torch.tensor(used_k, dtype=torch.int32, device=device)

    return seqlen_q, seqlen_k, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k


def gen_padding_mask_from_seqlens(seqlen, batch_size, seqused=None, device="musa"):
    max_seqlen = max(seqlen)
    lengths = torch.tensor(seqlen, dtype=torch.int32, device=device).reshape(
        batch_size, 1
    )
    if seqused is not None:
        lengths = torch.min(lengths, seqused.reshape(batch_size, 1))

    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size)
        < lengths
    )
    return padding_mask


def generate_random_padding_mask(
    max_seqlen, batch_size, device, mode="random", zero_lengths=False
):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full(
            (batch_size, 1), max_seqlen, device=device, dtype=torch.int32
        )
    elif mode == "random":
        lengths = torch.randint(
            max(0 if zero_lengths else 1, max_seqlen - 20),
            max_seqlen + 1,
            (batch_size, 1),
            device=device,
        )
    elif mode == "third":
        lengths = torch.randint(
            max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device
        )

    if zero_lengths:
        # Generate zero-lengths every 5 batches and the last batch.
        for i in range(batch_size):
            if i % 5 == 0:
                lengths[i] = 0
        lengths[-1] = 0
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size)
        < lengths
    )
    return padding_mask


def _rotate_half_torch(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2
        )


def apply_rotary_emb(
    x,
    cos,
    sin,
    interleaved=False,
    seqlen_offsets=0,
):
    """
    x: (B, S, H, D)
    cos, sin: (seqlen_ro, rotary_dim / 2)
    seqlen_offsets: int or (B,)
    return: (B, S, H, D)
    """
    assert x.dim() == 4
    B, S, H, D = x.shape
    seqlen_ro, rotary_half = cos.shape
    rotary_dim = rotary_half * 2
    assert sin.shape == cos.shape
    assert rotary_dim <= D

    if isinstance(seqlen_offsets, int):
        pos = torch.arange(S, device=x.device) + seqlen_offsets
        pos = pos.unsqueeze(0).expand(B, S)
    else:
        assert seqlen_offsets.shape == (B,)
        pos = torch.arange(S, device=x.device).unsqueeze(0) + seqlen_offsets.unsqueeze(
            1
        )

    assert pos.max().item() < seqlen_ro

    cos_pos = cos[pos].float()  # (B, S, rotary_dim / 2)
    sin_pos = sin[pos].float()  # (B, S, rotary_dim / 2)

    if not interleaved:
        cos_pos = repeat(cos_pos, "b s d -> b s 1 (2 d)")
        sin_pos = repeat(sin_pos, "b s d -> b s 1 (2 d)")
    else:
        cos_pos = repeat(cos_pos, "b s d -> b s 1 (d 2)")
        sin_pos = repeat(sin_pos, "b s d -> b s 1 (d 2)")

    x_ro = x[..., :rotary_dim].float()
    x_pass = x[..., rotary_dim:]

    out_ro = (
        x_ro * cos_pos + _rotate_half_torch(x_ro, interleaved=interleaved) * sin_pos
    ).to(x.dtype)
    return torch.cat([out_ro, x_pass], dim=-1)


def gen_input_tensor(
    batch_size,
    seqlens,
    num_head,
    headdim,
    use_cu_seqlens,
    datatype,
    device,
    randop=torch.randn,
):
    data = None
    cu_seqlens = None
    seqused = None

    if isinstance(seqlens, int):
        # bshd
        data = randop(
            batch_size, seqlens, num_head, headdim, device=device, dtype=datatype
        )

    elif isinstance(seqlens, list):
        # use cu_seqlen/seqused
        if use_cu_seqlens:
            total_seqlen = sum(seqlens)
            data = randop(
                total_seqlen, num_head, headdim, device=device, dtype=datatype
            )
            cu_seqlens = torch.tensor(
                [0] + seqlens, dtype=torch.int32, device=device
            ).cumsum(dim=0, dtype=torch.int32)
        else:
            data = randop(
                batch_size, seqlens, num_head, headdim, device=device, dtype=datatype
            )
            seqused = torch.tensor(seqlens, dtype=torch.int32, device=device)
    else:
        raise TypeError(f"seqlens must be either int or list, got {type(seqlens)}")

    return data, cu_seqlens, seqused


def gen_page_table(num_page, batch_size, device, mode="fa"):
    assert num_page % batch_size == 0
    page_per_batch = num_page // batch_size

    page_table = None
    if mode == "fa":
        page_table = rearrange(
            torch.randperm(num_page, dtype=torch.int32, device=device),
            "(b p) -> b p",
            b=batch_size,
        )
    elif mode == "arange":
        page_table = repeat(
            torch.arange(page_per_batch, dtype=torch.int32, device=device),
            "p -> b p",
            b=batch_size,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return page_table


def gen_kvcache_tensor(
    num_page,
    page_size,
    batch_size,
    max_seqlen_kv,
    head_kv,
    headdim_vo,
    datatype,
    device,
    page_mode="fa",
    init_page_table=None,
):
    assert max_seqlen_kv % page_size == 0

    min_page_per_batch = max_seqlen_kv // page_size
    min_page = min_page_per_batch * batch_size

    assert num_page >= min_page

    k_paged, _, _ = gen_input_tensor(
        batch_size=num_page,
        seqlens=page_size,
        num_head=head_kv,
        headdim=headdim_vo,
        use_cu_seqlens=False,
        datatype=datatype,
        device=device,
        randop=torch.randn,
    )
    v_paged, _, _ = gen_input_tensor(
        batch_size=num_page,
        seqlens=page_size,
        num_head=head_kv,
        headdim=headdim_vo,
        use_cu_seqlens=False,
        datatype=datatype,
        device=device,
        randop=torch.randn,
    )
    page_table = (
        gen_page_table(
            num_page=num_page, batch_size=batch_size, device=device, mode=page_mode
        )
        if init_page_table is None
        else init_page_table
    )

    assert page_table.shape[0] == batch_size
    assert page_table.shape[1] >= min_page_per_batch

    k = rearrange(
        k_paged[page_table.flatten()],
        "(b p) ps ... -> b (p ps) ...",
        b=batch_size,
    )[:, :max_seqlen_kv]

    v = rearrange(
        v_paged[page_table.flatten()],
        "(b p) ps ... -> b (p ps) ...",
        b=batch_size,
    )[:, :max_seqlen_kv]

    return k_paged, v_paged, page_table, k, v


def _mask_score(
    score,
    batch_idx,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    seqused_k: torch.Tensor = None,
):
    if (
        cu_seqlens_q is None
        and cu_seqlens_k is None
        and seqused_q is None
        and seqused_k is None
    ):
        return score[batch_idx], score.shape[2], score.shape[3]

    seqlen_q = _get_seqlen(
        batch_idx=batch_idx,
        seqused=seqused_q,
        cu_seqlens=cu_seqlens_q,
        seqlen=score.shape[2],
    )
    seqlen_k = _get_seqlen(
        batch_idx=batch_idx,
        seqused=seqused_k,
        cu_seqlens=cu_seqlens_k,
        seqlen=score.shape[3],
    )

    return score[batch_idx, :, :seqlen_q, :seqlen_k], seqlen_q, seqlen_k


def lse_ref_from_score(
    score: torch.Tensor,  # shape (batch, head, seqlen_q, seqlen_k)
    is_causal: bool = False,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_k: torch.Tensor = None,
    seqused_q: torch.Tensor = None,
    seqused_k: torch.Tensor = None,
    learnable_sink: Optional[torch.Tensor] = None,
):
    batch_size = score.shape[0]

    if learnable_sink is not None:
        learnable_sink = rearrange(learnable_sink, "h -> h 1")

    head_qo = score.shape[1]
    max_seqlen_q = score.shape[2]
    lse_ref_pad = torch.full((batch_size, head_qo, max_seqlen_q), float("-inf"))

    lse_refs = []
    for b in range(batch_size):
        s, seqlen_q, seqlen_k = _mask_score(
            score, b, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k
        )

        lse_ref = torch.logsumexp(s, dim=-1)
        if learnable_sink is not None:
            lse_ref = torch.logaddexp(lse_ref, learnable_sink)

        lse_ref = torch.nan_to_num(
            lse_ref,
            nan=float("-inf"),
            posinf=float("inf"),
            neginf=float("-inf"),
        )

        if is_causal and seqlen_q > seqlen_k:
            lse_ref[:, : seqlen_q - seqlen_k] = float("-inf")

        lse_ref_pad[b, :, :seqlen_q] = lse_ref
        lse_refs.append(lse_ref)

    lse_ref_unpad = torch.cat(lse_refs, dim=1)

    return lse_ref_unpad.to(score.device), lse_ref_pad.to(score.device)


class IndexFirstAxis(torch.autograd.Function):
    """
    Adapted from FlashAttention
    """

    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),
            0,
            repeat(indices, "z -> z d", d=second_dim),
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(
            0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    """
    Adapted from FlashAttention
    """

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (
        (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    )
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    if not is_fake_mode():
        indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
    else:
        batch_size, seqlen = attention_mask.shape
        indices = torch.arange(batch_size * seqlen, device=hidden_states.device)
        max_seqlen_in_batch = seqlen
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        IndexFirstAxis.apply(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Adapted from FlashAttention
    """
    output = IndexPutFirstAxis.apply(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def construct_chunk_mask(
    seqlen_q,
    seqlen_k,
    attention_chunk,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    """
    Adapted from FlashAttention
    """
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
    col_limit_left_chunk = row_idx + sk - sq - (row_idx + sk - sq) % attention_chunk
    return torch.logical_or(
        col_idx < col_limit_left_chunk,
        col_idx >= col_limit_left_chunk + attention_chunk,
    )


def generate_block_kvcache(
    max_seqlen_kv,
    page_size,
    batch_size,
    head_kv,
    headdim_qk,
    headdim_vo,
    device,
    dtype,
    dtype_ref=None,
    rand_op=torch.randn,
):
    """
    Adapted from FlashAttention
    """
    if dtype_ref is None:
        dtype_ref = dtype
    num_blocks = math.ceil(max_seqlen_kv / page_size) * batch_size * 3
    k_cache_paged = (
        rand_op(
            num_blocks, page_size, head_kv, headdim_qk, device=device, dtype=dtype_ref
        )
        .to(dtype)
        .to(dtype_ref)
    )
    v_cache_paged = (
        rand_op(
            num_blocks, page_size, head_kv, headdim_vo, device=device, dtype=dtype_ref
        )
        .to(dtype)
        .to(dtype_ref)
    )
    page_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device=device),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    k_cache = rearrange(
        k_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :max_seqlen_kv]
    v_cache = rearrange(
        v_cache_paged[page_table.flatten()],
        "(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :max_seqlen_kv]

    return k_cache, v_cache, page_table, k_cache_paged, v_cache_paged, num_blocks


def arange_along_dim(
    tensor: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    assert 0 <= dim < tensor.dim(), (
        f"Dimension {dim} out of range for tensor with {tensor.dim()} dimensions"
    )
    return (
        torch.arange(tensor.size(dim), device=tensor.device, dtype=tensor.dtype)
        .view(*[1 if i != dim else -1 for i in range(tensor.dim())])
        .expand(*tensor.shape)
    )


def address_to_tensor_coords(
    tensor: torch.Tensor,
    address: int,
) -> Tuple[int, ...]:
    """
    Convert a memory address to tensor coordinates.

    Args:
        tensor:
            A PyTorch tensor.
        address:
            Target memory address (integer).

    Returns:
        Tuple of coordinates corresponding to the element.

    Raises:
        ValueError:
            If the address is outside the tensor storage range or
            does not align to an element boundary.
    """

    # Base address of tensor storage
    base_addr = tensor.data_ptr()

    # Element size in bytes
    elem_size = tensor.element_size()

    # Tensor metadata
    shape = tensor.shape
    strides = tensor.stride()

    # Compute valid address range
    max_offset = 0
    for dim_size, stride in zip(shape, strides):
        if dim_size > 0:
            max_offset += (dim_size - 1) * stride

    storage_bytes = (max_offset + 1) * elem_size
    end_addr = base_addr + storage_bytes

    # Range check
    if not (base_addr <= address < end_addr):
        raise ValueError(
            f"Address 0x{address:x} is outside tensor range "
            f"[0x{base_addr:x}, 0x{end_addr:x})"
        )

    # Byte offset from tensor base
    byte_offset = address - base_addr

    # Must align to element boundary
    if byte_offset % elem_size != 0:
        raise ValueError(
            f"Address 0x{address:x} is not aligned to element size ({elem_size} bytes)"
        )

    # Convert to element offset
    elem_offset = byte_offset // elem_size

    # Recover coordinates using strides
    coords = []
    remaining = elem_offset

    for dim_size, stride in zip(shape, strides):
        coord = remaining // stride
        remaining = remaining % stride

        if coord >= dim_size:
            raise ValueError("Address does not map to a valid tensor coordinate")

        coords.append(coord)

    return tuple(coords)


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(None, None),
    sink_token_length=0,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    device=None,
):
    """
    Adapted from FlashAttention
    """
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] is None:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        if window_size[1] is None:
            local_mask_left = col_idx > sk
        else:
            local_mask_left = col_idx > torch.minimum(
                row_idx + sk - sq + window_size[1], sk
            )
        return torch.logical_or(
            local_mask_left,
            torch.logical_and(
                col_idx < row_idx + sk - sq - window_size[0],
                col_idx >= sink_token_length,
            ),
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    key_leftpad=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(None, None),
    attention_chunk=0,
    sink_token_length=0,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    intermediate_dtype=None,
    only_qv=False,
):
    """
    Adapted from FlashAttention
    """
    pass
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
        qv = qv.float() if qv is not None else None
    if q_descale is not None:
        q_descale = repeat(q_descale, "b h -> b 1 (h g) 1", g=q.shape[2] // k.shape[2])
        q = (q.float() * q_descale).to(q.dtype)
        qv = (qv.float() * q_descale).to(qv.dtype) if qv is not None else None
    if k_descale is not None:
        k = (k.float() * rearrange(k_descale, "b h -> b 1 h 1")).to(dtype=k.dtype)
    if v_descale is not None:
        v = (v.float() * rearrange(v_descale, "b h -> b 1 h 1")).to(dtype=v.dtype)
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    dv = v.shape[-1]
    softmax_scale = 1.0 / math.sqrt(dv if only_qv else d if qv is None else d + dv)
    if only_qv:
        assert qv is not None
        scores = torch.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
    elif not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if qv is not None and not only_qv:
        scores = scores + torch.einsum("bthd,bshd->bhts", qv * softmax_scale, v)
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    local_mask = None
    if window_size[0] is not None or window_size[1] is not None:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            sink_token_length,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
    if attention_chunk > 0:
        chunk_mask = construct_chunk_mask(
            seqlen_q,
            seqlen_k,
            attention_chunk,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
            device=q.device,
        )
        local_mask = (
            torch.logical_or(local_mask, chunk_mask)
            if local_mask is not None
            else chunk_mask
        )
    if local_mask is not None:
        scores.masked_fill_(local_mask, float("-inf"))
    if attn_bias is not None:
        scores = scores + attn_bias
    if learnable_sink is None:
        attention = torch.softmax(scores, dim=-1).to(v.dtype)
    else:
        scores_fp32 = scores.to(torch.float32)
        logits_max = torch.amax(scores_fp32, dim=-1, keepdim=True)
        learnable_sink = rearrange(learnable_sink, "h -> h 1 1")
        logits_or_sinks_max = torch.maximum(learnable_sink, logits_max)
        unnormalized_scores = torch.exp(scores_fp32 - logits_or_sinks_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + torch.exp(
            learnable_sink - logits_or_sinks_max
        )
        attention = (unnormalized_scores / normalizer).to(v.dtype)
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    if key_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0
        )
    if local_mask is not None:
        attention = attention.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    if intermediate_dtype is not None:
        attention_drop = attention_drop.to(intermediate_dtype).to(attention_drop.dtype)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og), scores


def attention_accum_ref(
    query: torch.Tensor,  # bshd
    key: torch.Tensor,  # bshd
    value: torch.Tensor,  # bshd
    metadata: torch.Tensor,
    num_splits=0,
    softmax_scale=None,
    seqused_q=None,
    seqused_kv=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    causal=False,
    window_size=(None, None),
    sink_token_length=0,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_n=64,
):
    # TODO: unsupported
    assert causal is False
    assert window_size == (None, None)
    assert sink_token_length == 0
    assert learnable_sink is None

    device = query.device
    batch_size = query.shape[0]

    seqlen_q = query.shape[1]
    head_q = query.shape[2]
    headdim_q = query.shape[3]

    softmax_scale = (
        1.0 / math.sqrt(headdim_q) if softmax_scale is None else softmax_scale
    )

    assert key.shape[0] == batch_size
    assert value.shape[0] == batch_size
    seqlen_kv = key.shape[1]
    # head_kv = value.shape[2]

    metadata = metadata.view(4, -1)
    num_splits = torch.max(metadata[0]).item() if num_splits == 0 else num_splits

    assert seqused_q is None or seqused_q.shape[0] == batch_size
    assert seqused_kv is None or seqused_kv.shape[0] == batch_size
    assert cu_seqlens_q is None or cu_seqlens_q.shape[0] == batch_size + 1
    assert cu_seqlens_kv is None or cu_seqlens_kv.shape[0] == batch_size + 1

    ref_o_accum = torch.zeros(
        (num_splits, batch_size, head_q, seqlen_q, headdim_q),
        dtype=torch.float,
        device=device,
    )
    ref_lse_accum = torch.zeros(
        (num_splits, batch_size, head_q, seqlen_q), dtype=torch.float, device=device
    )

    # print(
    #     f"batch: {batch_size} seqlen_q: {seqlen_q} seqlen_kv: {seqlen_kv} seqused_kv: {seqused_kv} cu_seqlens_q: {cu_seqlens_q}"
    # )

    for b in range(batch_size):
        real_batch_idx = metadata[1, b]

        real_seqlen_q = _get_seqlen(real_batch_idx, seqused_q, cu_seqlens_q, seqlen_q)
        real_seqlen_kv = _get_seqlen(
            real_batch_idx, seqused_kv, cu_seqlens_kv, seqlen_kv
        )

        real_splits = metadata[0, b]
        # num_blk_per_tile = ceil_div(seqlen_kv, tile_n)
        # num_tile_per_split = ceil_div(num_blk_per_tile, real_splits)

        if real_splits <= 1:
            continue
        kv_len_per_split = ceil_div(real_seqlen_kv, real_splits)

        # print(f"Batch {b}, seqlen_q: {real_seqlen_q}, seqlen_kv: {real_seqlen_kv}, splits: {real_splits}")

        q = query[real_batch_idx, :real_seqlen_q, :, :]

        for split in range(real_splits):
            k = key[
                real_batch_idx,
                split * kv_len_per_split : (split + 1) * kv_len_per_split,
                :,
                :,
            ]
            v = value[
                real_batch_idx,
                split * kv_len_per_split : (split + 1) * kv_len_per_split,
                :,
                :,
            ]

            score = torch.einsum("thd,shd->hts", q * softmax_scale, k).to(torch.float)
            attn = torch.softmax(score, dim=-1).to(v.dtype)

            lse = torch.logsumexp(score, dim=-1)
            output = torch.einsum("hts,shd->htd", attn, v).to(torch.float)

            ref_o_accum[split, real_batch_idx, :, :real_seqlen_q, :] = output
            ref_lse_accum[split, real_batch_idx, :, :real_seqlen_q] = lse

    return ref_o_accum, ref_lse_accum


def attention_combine_ref(
    out_accum: torch.Tensor,
    lse_accum: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates reference combine result for split-kv attention.

    Parameters
    ----------
    out_accum : torch.Tensor
        (split, batch, head, seqlen_q, dv) / (split, 1, head, total_q, dv)
    lse_accum : torch.Tensor
        (split, batch, head, seqlen_q) / (split, 1, head, total_q)

    Returns
    -------
    out : torch.Tensor
        (batch, head, seqlen_q, dv) or (1, head, total_q, dv)
    lse : torch.Tensor
        (batch, head, seqlen_q) or (1, head, total_q)
    """
    lse = torch.logsumexp(lse_accum, dim=0)
    scale = torch.exp(lse_accum - lse.unsqueeze(0))
    scale = torch.where(
        torch.isinf(scale) | torch.isnan(scale), torch.zeros_like(scale), scale
    )
    out = (scale.unsqueeze(-1) * out_accum).sum(dim=0).transpose(1, 2)
    return out, lse


def pad_accum(
    out_accum,
    lse_accum,
    seqused_q=None,
    cu_seqlens_q=None,
):
    assert seqused_q is not None or cu_seqlens_q is not None

    if seqused_q is not None:
        batch_size = seqused_q.shape[0]
        max_seqlen = torch.max(seqused_q).item()
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        max_seqlen = max(
            [cu_seqlens_q[i + 1] - cu_seqlens_q[i] for i in range(batch_size)]
        )

    head = out_accum.shape[1]
    num_splits = out_accum.shape[2]
    headdim = out_accum.shape[4]

    o_accum_pad = torch.zeros(
        (batch_size, head, num_splits, max_seqlen, headdim),
        dtype=out_accum.dtype,
        device=out_accum.device,
    )
    lse_accum_pad = torch.zeros(
        (batch_size, head, num_splits, max_seqlen),
        dtype=out_accum.dtype,
        device=out_accum.device,
    )
    seqlen_base = 0
    for b in range(batch_size):
        seqlen = _get_seqlen(b, seqused_q, cu_seqlens_q, max_seqlen)
        o_accum_pad[b, :, :, :seqlen, :] = out_accum[
            0, :, :, seqlen_base : seqlen_base + seqlen, :
        ]
        lse_accum_pad[b, :, :, :seqlen] = lse_accum[
            0, :, :, seqlen_base : seqlen_base + seqlen
        ]
        seqlen_base += seqlen

    return o_accum_pad, lse_accum_pad


def _combine_cp_partials(
    out_list: list,
    lse_list: list,
    lse_is_log2: bool = False,
):
    """Online-softmax combine for Context Parallelism partial outputs.

    Each rank computes attention over its local KV slice. This function
    aggregates the partial results using the standard log-sum-exp trick,
    equivalent to computing attention over the full KV sequence.

    Args:
        out_list: List of per-rank output tensors, each [seqlen_q, nhead, headdim_v], float32.
        lse_list: List of per-rank log-sum-exp tensors, each [nhead, seqlen_q], float32.
        lse_is_log2: If True, lse values are in log base-2 (e.g. from FlashAttention2);
                     they will be converted to natural log before combining.

    Returns:
        out_combined: [seqlen_q, nhead, headdim_v], float32.
        lse_combined: [nhead, seqlen_q], float32.
    """
    if lse_is_log2:
        lse_list = [x * math.log(2.0) for x in lse_list]

    lse_combined = torch.stack(lse_list, dim=0).logsumexp(dim=0)  # [nh, sq]
    out_combined = torch.zeros_like(out_list[0])
    for out_r, lse_r in zip(out_list, lse_list):
        scale = (lse_r - lse_combined).exp().T.unsqueeze(-1)  # [sq, nh, 1]
        out_combined += scale * out_r
    return out_combined, lse_combined


def make_cp_rank_local_paged_kvcache(
    k_cache_paged: torch.Tensor,
    v_cache_paged: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    page_size: int,
    cp_rank: int,
    cp_world_size: int,
    device: str,
):
    """Simulate framework-side KV cache preparation for CP + paged KV attention.

    In production (vLLM/SGLang), the framework organizes the KV cache so that
    each CP rank receives a rank-local paged KV cache before calling the
    attention kernel. The kernel itself is CP-unaware for paged KV; CP semantics
    are handled solely in seqlen/mask/block_info via cp_rank/cp_world_size/
    cp_tot_seqused_k.

    This function simulates that framework behavior for testing purposes:
    it gathers rank-local tokens from the global paged KV cache using token-level
    interleaving (interleave_size=1, the default in vLLM), then repacks them
    into a fresh rank-local paged KV cache with a randomized page table.

    Token-level interleaving: rank r sees global tokens r, r+W, r+2W, ...
    where W = cp_world_size. The mask formula local_j * W + r then gives the
    correct global token position for causal masking.

    Note: Only interleave_size=1 is supported, consistent with FA and vLLM defaults.

    Args:
        k_cache_paged: Global K cache, shape (num_blocks, page_size, head_kv, headdim_qk).
        v_cache_paged: Global V cache, shape (num_blocks, page_size, head_kv, headdim_vo).
        page_table:    Global page table, shape (batch, max_pages_per_seq), int32.
        cache_seqlens: Global KV sequence lengths, shape (batch,), int32.
        page_size:     Number of tokens per page.
        cp_rank:       Rank index of this CP worker [0, cp_world_size).
        cp_world_size: Total number of CP ranks.
        device:        Target device string.

    Returns:
        k_local_paged:       Rank-local K cache, shape (total_local_pages, page_size, head_kv, headdim_qk).
        v_local_paged:       Rank-local V cache, shape (total_local_pages, page_size, head_kv, headdim_vo).
        page_table_local:    Rank-local page table, shape (batch, max_local_pages), int32.
        local_seqlens_tensor: Rank-local KV sequence lengths, shape (batch,), int32.
    """
    batch_size = page_table.shape[0]
    head_kv = k_cache_paged.shape[2]
    headdim_qk = k_cache_paged.shape[3]
    headdim_vo = v_cache_paged.shape[3]

    # Step 1: revert paged KV to dense (batch, max_seqlen, head, dim)
    max_pages = page_table.shape[1]
    flat_pages = page_table.flatten()
    k_dense = k_cache_paged[flat_pages].view(
        batch_size, max_pages * page_size, head_kv, headdim_qk
    )
    v_dense = v_cache_paged[flat_pages].view(
        batch_size, max_pages * page_size, head_kv, headdim_vo
    )

    # Step 2: compute rank-local seqlen per batch item
    local_seqlens = []
    if is_fake_mode():
        local_seqlens = [0 for _ in range(batch_size)]
    else:
        for b in range(batch_size):
            sk = cache_seqlens[b].item()
            local_len = len(range(cp_rank, sk, cp_world_size))
            local_seqlens.append(local_len)

    max_local_seqlen = max(local_seqlens) if local_seqlens else 0
    max_local_pages = (
        math.ceil(max_local_seqlen / page_size) if max_local_seqlen > 0 else 1
    )
    total_local_pages = (
        max_local_pages * batch_size * 3
    )  # 3x headroom for random allocation

    k_local_paged = torch.zeros(
        total_local_pages,
        page_size,
        head_kv,
        headdim_qk,
        device=device,
        dtype=k_cache_paged.dtype,
    )
    v_local_paged = torch.zeros(
        total_local_pages,
        page_size,
        head_kv,
        headdim_vo,
        device=device,
        dtype=v_cache_paged.dtype,
    )
    page_table_local = torch.zeros(
        batch_size, max_local_pages, dtype=torch.int32, device=device
    )

    all_page_indices = torch.randperm(total_local_pages, device=device)
    page_cursor = 0

    for b in range(batch_size):
        if is_fake_mode():
            break
        sk = cache_seqlens[b].item()
        local_indices = torch.arange(cp_rank, sk, cp_world_size, device=device)
        local_len = local_indices.shape[0]
        n_pages = math.ceil(local_len / page_size) if local_len > 0 else 0

        if local_len == 0:
            continue

        k_local_tokens = k_dense[b][local_indices]
        v_local_tokens = v_dense[b][local_indices]

        for p in range(n_pages):
            pg_idx = all_page_indices[page_cursor].item()
            page_cursor += 1
            page_table_local[b, p] = pg_idx

            start = p * page_size
            end = min(start + page_size, local_len)
            k_local_paged[pg_idx, : end - start] = k_local_tokens[start:end]
            v_local_paged[pg_idx, : end - start] = v_local_tokens[start:end]

    local_seqlens_tensor = torch.tensor(local_seqlens, dtype=torch.int32, device=device)
    return k_local_paged, v_local_paged, page_table_local, local_seqlens_tensor
