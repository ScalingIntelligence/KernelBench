from __future__ import annotations

import pytest
import torch

import mate
from mate import msa_interface as msa

_HAS_MUSA = hasattr(torch, "musa") and torch.musa.is_available()
_HAS_FP8_E4M3 = hasattr(torch, "float8_e4m3fn")
_FWD_DTYPES = [torch.float16, torch.bfloat16]
if _HAS_FP8_E4M3:
    _FWD_DTYPES.append(torch.float8_e4m3fn)


def test_msa_maxscore_prefill_tile_q_dispatch():
    from mate.jit.msa_ops import _maxscore_tile_q

    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.bfloat16,
            max_seqlen_q=64 * 1024,
        )
        == 128
    )
    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.bfloat16,
            max_seqlen_q=32 * 1024,
        )
        == 16
    )
    assert (
        _maxscore_tile_q(
            1,
            True,
            dtype=torch.float8_e4m3fn if _HAS_FP8_E4M3 else torch.bfloat16,
            max_seqlen_q=128 * 1024,
        )
        == 16
    )


def _dense_sparse_prefill_reference(
    q: torch.Tensor,
    k_pages_hpd: torch.Tensor,
    v_pages_hpd: torch.Tensor,
    page_table: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    *,
    causal: bool,
    softmax_scale: float,
    qo_offset: int | None = None,
    q_lens: torch.Tensor | None = None,
    kv_lens: torch.Tensor | None = None,
    sparse_block_size: int = 128,
    upcast: bool = True,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    num_pages, num_kv_heads, page_size, head_dim_qk = k_pages_hpd.shape
    _, _, _, head_dim_v = v_pages_hpd.shape
    total_q, num_qo_heads, _ = q.shape
    qhead_per_kv = num_qo_heads // num_kv_heads
    batch_size = int(page_table.shape[0])
    if q_lens is None:
        if batch_size != 1:
            raise ValueError("q_lens is required for multi-batch reference inputs")
        q_lens_cpu = [total_q]
    else:
        q_lens_cpu = [int(v) for v in q_lens.cpu().tolist()]
    if kv_lens is None:
        kv_lens_cpu = [int(page_table.shape[1]) * page_size] * batch_size
    else:
        kv_lens_cpu = [int(v) for v in kv_lens.cpu().tolist()]
    if sum(q_lens_cpu) != total_q:
        raise ValueError(f"q_lens sum mismatch: {sum(q_lens_cpu)} vs {total_q}")

    out_dtype = q.dtype if output_dtype is None else output_dtype
    out = torch.zeros(
        (total_q, num_qo_heads, head_dim_v),
        dtype=out_dtype,
        device=q.device,
    )
    q_abs = 0
    for batch_idx, q_len in enumerate(q_lens_cpu):
        kv_len = kv_lens_cpu[batch_idx]
        q_offset = qo_offset if qo_offset is not None else kv_len - q_len
        for q_local in range(q_len):
            for head in range(num_kv_heads):
                hq_begin = head * qhead_per_kv
                hq_end = hq_begin + qhead_per_kv
                k_chunks = []
                v_chunks = []
                for kv_block_idx in kv_block_indexes[q_abs, head].tolist():
                    if kv_block_idx < 0:
                        continue
                    block_start = int(kv_block_idx) * sparse_block_size
                    if block_start >= kv_len:
                        continue
                    block_end = min(block_start + sparse_block_size, kv_len)
                    if causal:
                        block_end = min(block_end, q_local + q_offset + 1)
                    if block_end <= block_start:
                        continue
                    cursor = block_start
                    while cursor < block_end:
                        logical_page = cursor // page_size
                        page_begin = cursor - logical_page * page_size
                        chunk_end = min(block_end, (logical_page + 1) * page_size)
                        page_end = chunk_end - logical_page * page_size
                        physical_page = int(page_table[batch_idx, logical_page].item())
                        if physical_page < 0 or physical_page >= num_pages:
                            raise ValueError(f"invalid physical page {physical_page}")
                        k_chunks.append(
                            k_pages_hpd[physical_page, head, page_begin:page_end]
                        )
                        v_chunks.append(
                            v_pages_hpd[physical_page, head, page_begin:page_end]
                        )
                        cursor = chunk_end
                if not k_chunks:
                    continue
                k_sel = torch.cat(k_chunks, dim=0)
                v_sel = torch.cat(v_chunks, dim=0)
                q_sel = q[q_abs, hq_begin:hq_end]
                if upcast:
                    k_sel = k_sel.to(torch.float32)
                    v_sel = v_sel.to(torch.float32)
                    q_sel = q_sel.to(torch.float32)
                scores = torch.matmul(q_sel, k_sel.transpose(0, 1)) * softmax_scale
                probs = torch.softmax(scores, dim=-1)
                out[q_abs, hq_begin:hq_end] = torch.matmul(probs, v_sel).to(
                    dtype=out_dtype
                )
            q_abs += 1
    return out


def _msa_output_maxscore_k_tiles(max_kv_len: int) -> int:
    kv_tiles = (int(max_kv_len) + 127) // 128
    return ((kv_tiles + 127) // 128) * 128


def _dense_msa_output_maxscore_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    qo_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    num_kv_heads: int,
    max_k_tiles: int,
    *,
    causal: bool,
    qo_offset: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    page_size: int | None = None,
) -> torch.Tensor:
    total_q, num_qo_heads, _ = q.shape
    qhead_per_kv = num_qo_heads // num_kv_heads
    result = torch.full(
        (total_q, num_qo_heads, max_k_tiles),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    q_lens_cpu = [int(v) for v in qo_lens.cpu().tolist()]
    kv_lens_cpu = [int(v) for v in kv_lens.cpu().tolist()]
    q_abs = 0
    kv_abs = 0
    for batch_idx, (q_len, kv_len) in enumerate(zip(q_lens_cpu, kv_lens_cpu)):
        q_seq = q[q_abs : q_abs + q_len].float()
        if page_table is None:
            k_seq = k[kv_abs : kv_abs + kv_len].float()
        else:
            assert page_size is not None
            pages = []
            for logical_page in range((kv_len + page_size - 1) // page_size):
                physical_page = int(page_table[batch_idx, logical_page].item())
                page_begin = logical_page * page_size
                page_end = min(page_begin + page_size, kv_len)
                pages.append(
                    k[physical_page, :, : page_end - page_begin]
                    .permute(1, 0, 2)
                    .float()
                )
            k_seq = torch.cat(pages, dim=0) if pages else k.new_empty((0, 0, 0))
        causal_off = (
            int(qo_offset[batch_idx].item())
            if qo_offset is not None
            else kv_len - q_len
        )
        for tile_idx in range((kv_len + 127) // 128):
            k_begin = tile_idx * 128
            k_end = min(k_begin + 128, kv_len)
            k_tile = k_seq[k_begin:k_end].repeat_interleave(qhead_per_kv, dim=1)
            scores = torch.einsum("qhd,khd->qhk", q_seq, k_tile)
            if causal:
                q_pos = torch.arange(q_len, device=q.device).unsqueeze(1) + causal_off
                k_pos = torch.arange(k_begin, k_end, device=q.device).unsqueeze(0)
                valid_mask = q_pos >= k_pos
                scores = scores.masked_fill(~valid_mask.unsqueeze(1), -float("inf"))
                tile_max = scores.max(dim=-1).values
                tile_max = torch.where(
                    valid_mask.any(dim=1).unsqueeze(1),
                    tile_max,
                    torch.full_like(tile_max, -float("inf")),
                )
            else:
                tile_max = scores.max(dim=-1).values
            result[q_abs : q_abs + q_len, :, tile_idx] = tile_max
        q_abs += q_len
        kv_abs += kv_len
    return result


def _assert_maxscore_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual[finite],
        expected[finite],
        atol=1e-3,
        rtol=1e-3,
    )
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))


def _assert_msa_prefill_forward_close(
    actual: torch.Tensor,
    ref_fp32: torch.Tensor,
    ref_low_precision: torch.Tensor,
) -> None:
    fwd_atol = 2 * (ref_fp32 + 0.3 - 0.3 - ref_fp32).abs().max().item()
    pt_diff = (ref_low_precision.float() - ref_fp32).abs().max().item()
    kernel_diff = (actual.float() - ref_fp32).abs().max().item()
    allowed = 1.25 * (2 * pt_diff + fwd_atol)
    if actual.dtype == torch.float16:
        allowed = max(allowed, 5e-3)
    assert kernel_diff <= allowed, (
        "MSA prefill tolerance failed: "
        f"kernel_diff={kernel_diff:.8g}, pt_diff={pt_diff:.8g}, "
        f"fwd_atol={fwd_atol:.8g}, allowed={allowed:.8g}"
    )


def _assert_msa_regression_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    threshold = 0.99999 if dtype == torch.bfloat16 else 0.9995
    threshold_diff = 0.02 if dtype == torch.bfloat16 else 0.11
    actual_f = actual.float()
    ref_f = ref.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        actual_f.reshape(-1), ref_f.reshape(-1), dim=0
    ).item()
    max_diff = (actual_f - ref_f).abs().max().item()
    assert cos_sim > threshold and max_diff < threshold_diff, (
        "MSA regression tolerance failed: "
        f"cos_sim={cos_sim:.8g} threshold={threshold:.8g}, "
        f"max_diff={max_diff:.8g} threshold_diff={threshold_diff:.8g}"
    )


def _small_randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    scale: float = 1.0,
) -> torch.Tensor:
    return (torch.randn(shape, dtype=torch.float32) * scale).to(
        device=device, dtype=dtype
    )


def _page_counts_from_lens(kv_lens: torch.Tensor, page_size: int) -> list[int]:
    return [int((int(length) + page_size - 1) // page_size) for length in kv_lens]


def _make_page_table_and_kv_indices(
    kv_lens: torch.Tensor,
    *,
    page_size: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_device = kv_lens.device
    page_counts = _page_counts_from_lens(kv_lens.cpu(), page_size)
    max_pages = max(page_counts, default=0)
    total_pages = sum(page_counts)
    if mode == "identity":
        physical_pages = list(range(total_pages))
    elif mode == "reverse":
        physical_pages = list(reversed(range(total_pages)))
    else:
        raise ValueError(f"unsupported page table mode: {mode}")

    page_table = torch.zeros(
        (len(page_counts), max_pages), dtype=torch.int32, device=output_device
    )
    flat_indices = []
    cursor = 0
    for batch_idx, count in enumerate(page_counts):
        pages = physical_pages[cursor : cursor + count]
        if pages:
            page_table[batch_idx, :count] = torch.tensor(
                pages, dtype=torch.int32, device=output_device
            )
            flat_indices.extend(pages)
        cursor += count
    return page_table, torch.tensor(
        flat_indices, dtype=torch.int32, device=output_device
    )


def _select_sparse_blocks(
    *,
    pattern: str,
    q_local: int,
    page_count: int,
    topk: int,
) -> list[int]:
    if page_count <= 0:
        return []
    if pattern == "first":
        selected = list(range(page_count))
    elif pattern == "tail":
        selected = list(range(max(0, page_count - topk), page_count))
    elif pattern in {"rolling", "rolling_holes"}:
        selected = []
        cursor = q_local % page_count
        while len(selected) < min(topk, page_count):
            if cursor not in selected:
                selected.append(cursor)
            cursor = (cursor + 1) % page_count
    elif pattern == "sink_local":
        middle_count = max(0, topk - 2)
        middle = list(range(1, max(1, page_count - 1)))[:middle_count]
        selected = [0, *middle]
        if page_count > 1:
            selected.append(page_count - 1)
    else:
        raise ValueError(f"unsupported sparse block pattern: {pattern}")
    selected = selected[: min(topk, page_count)]
    if pattern == "rolling_holes" and q_local % 3 == 0 and len(selected) > 1:
        selected = selected[:-1]
    return sorted(selected)


def _make_kv_block_indexes(
    q_lens: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    topk: int,
    sparse_block_size: int,
    pattern: str,
    device: torch.device,
) -> torch.Tensor:
    q_lens_cpu = [int(v) for v in q_lens.cpu().tolist()]
    kv_lens_cpu = [int(v) for v in kv_lens.cpu().tolist()]
    total_q = sum(q_lens_cpu)
    q2k = torch.full(
        (total_q, num_kv_heads, topk),
        -1,
        dtype=torch.int32,
        device=device,
    )
    q_abs = 0
    for q_len, kv_len in zip(q_lens_cpu, kv_lens_cpu):
        page_count = (int(kv_len) + sparse_block_size - 1) // sparse_block_size
        for q_local in range(q_len):
            selected = _select_sparse_blocks(
                pattern=pattern,
                q_local=q_local,
                page_count=page_count,
                topk=topk,
            )
            if selected:
                q2k[q_abs, :, : len(selected)] = torch.tensor(
                    selected,
                    dtype=torch.int32,
                    device=device,
                ).view(1, len(selected))
            q_abs += 1
    return q2k.contiguous()


def _make_sparse_integration_inputs(
    case: dict[str, object],
    *,
    device: torch.device,
) -> dict[str, torch.Tensor | int | bool | torch.dtype]:
    page_size = int(case.get("page_size", 128))
    sparse_block_size = int(case.get("sparse_block_size", 128))
    head_dim = 128
    q_lens = torch.tensor(case["q_lens"], dtype=torch.int32, device=device)
    kv_lens = torch.tensor(case["kv_lens"], dtype=torch.int32, device=device)
    dtype = case["dtype"]
    assert isinstance(dtype, torch.dtype)
    q_heads = int(case["q_heads"])
    kv_heads = int(case["kv_heads"])
    topk = int(case["topk"])
    page_table, kv_indices = _make_page_table_and_kv_indices(
        kv_lens,
        page_size=page_size,
        mode=str(case["page_mode"]),
    )
    total_q = int(q_lens.sum().item())
    total_pages = int(kv_indices.numel())

    q = _small_randn((total_q, q_heads, head_dim), device=device, dtype=dtype)
    k_pages = _small_randn(
        (total_pages, kv_heads, page_size, head_dim),
        device=device,
        dtype=dtype,
    )
    v_pages = _small_randn(
        (total_pages, kv_heads, page_size, head_dim),
        device=device,
        dtype=dtype,
    )
    kv_block_indexes = _make_kv_block_indexes(
        q_lens,
        kv_lens,
        num_kv_heads=kv_heads,
        topk=topk,
        sparse_block_size=sparse_block_size,
        pattern=str(case["pattern"]),
        device=device,
    )
    return {
        "q": q,
        "k_pages": k_pages,
        "v_pages": v_pages,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "kv_indices": kv_indices,
        "kv_block_indexes": kv_block_indexes,
        "page_table": page_table,
        "page_size": page_size,
        "sparse_block_size": sparse_block_size,
        "head_dim": head_dim,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "topk": topk,
        "dtype": dtype,
        "causal": bool(case["causal"]),
    }


_SPARSE_PREFILL_INTEGRATION_CASES = [
    {
        "name": "p01_fp16_b1_first_t4",
        "q_lens": [64],
        "kv_lens": [128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "identity",
    },
    {
        "name": "p02_fp16_b1_rolling_t8_partial",
        "q_lens": [64],
        "kv_lens": [192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "identity",
    },
    {
        "name": "p03_fp16_b1_tail_reverse",
        "q_lens": [96],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
    },
    {
        "name": "p04_fp16_b2_rolling_reverse",
        "q_lens": [48, 64],
        "kv_lens": [128, 192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "p05_fp16_b2_holes_t16",
        "q_lens": [64, 64],
        "kv_lens": [256, 128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 16,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling_holes",
        "page_mode": "identity",
    },
    {
        "name": "p06_fp16_noncausal_reverse",
        "q_lens": [96],
        "kv_lens": [192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": False,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "p07_fp16_hkv2_first",
        "q_lens": [64],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "identity",
    },
    {
        "name": "p08_fp16_hkv2_b2_tail",
        "q_lens": [48, 64],
        "kv_lens": [192, 256],
        "q_heads": 8,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
    },
    {
        "name": "p09_bf16_first_reverse",
        "q_lens": [64],
        "kv_lens": [128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.bfloat16,
        "causal": True,
        "pattern": "first",
        "page_mode": "reverse",
    },
    {
        "name": "p10_bf16_b2_rolling",
        "q_lens": [64, 48],
        "kv_lens": [256, 192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.bfloat16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "identity",
    },
    {
        "name": "p11_fp16_hq16_first",
        "q_lens": [64],
        "kv_lens": [256],
        "q_heads": 16,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "identity",
    },
    {
        "name": "p12_fp16_hq16_holes_t16",
        "q_lens": [64],
        "kv_lens": [256],
        "q_heads": 16,
        "kv_heads": 2,
        "topk": 16,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling_holes",
        "page_mode": "reverse",
    },
    {
        "name": "p12b_fp16_t16_plus_sink_local",
        "q_lens": [64],
        "kv_lens": [4096],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 18,
        "selection_topk": 16,
        "force_begin_blocks": 1,
        "force_end_blocks": 1,
        "force_blocks_count_in_topk": False,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "sink_local",
        "page_mode": "identity",
    },
    {
        "name": "p13_fp16_q80_tail",
        "q_lens": [80],
        "kv_lens": [128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "identity",
    },
    {
        "name": "p14_fp16_b2_small_prefill",
        "q_lens": [32, 40],
        "kv_lens": [128, 128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "reverse",
    },
    {
        "name": "p15_bf16_hkv2_noncausal",
        "q_lens": [64],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.bfloat16,
        "causal": False,
        "pattern": "rolling",
        "page_mode": "identity",
    },
    {
        "name": "p16_fp16_q128_t16_reverse",
        "q_lens": [128],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 16,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "p17_fp16_page128_block128_tail",
        "q_lens": [64],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
        "page_size": 128,
        "sparse_block_size": 128,
    },
]

_SPARSE_DECODE_INTEGRATION_CASES = [
    {
        "name": "d01_fp16_q1_first",
        "q_lens": [1],
        "kv_lens": [128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "identity",
    },
    {
        "name": "d02_fp16_q4_rolling_reverse",
        "q_lens": [4],
        "kv_lens": [128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "d03_fp16_b2_first_partial",
        "q_lens": [2, 4],
        "kv_lens": [128, 192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "identity",
    },
    {
        "name": "d04_fp16_b2_rolling_t8",
        "q_lens": [4, 4],
        "kv_lens": [256, 128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "d05_fp16_b2_tail_t8",
        "q_lens": [1, 3],
        "kv_lens": [192, 192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "identity",
    },
    {
        "name": "d06_fp16_q4_tail_t16",
        "q_lens": [4],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 16,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
    },
    {
        "name": "d07_fp16_hkv2_first",
        "q_lens": [4],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "identity",
    },
    {
        "name": "d08_fp16_hkv2_b2_rolling",
        "q_lens": [2, 4],
        "kv_lens": [192, 256],
        "q_heads": 8,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "d09_bf16_q4_first",
        "q_lens": [4],
        "kv_lens": [128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.bfloat16,
        "causal": True,
        "pattern": "first",
        "page_mode": "reverse",
    },
    {
        "name": "d10_bf16_b2_rolling",
        "q_lens": [1, 4],
        "kv_lens": [256, 128],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.bfloat16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "identity",
    },
    {
        "name": "d11_fp16_hq16_tail",
        "q_lens": [4],
        "kv_lens": [256],
        "q_heads": 16,
        "kv_heads": 2,
        "topk": 16,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
    },
    {
        "name": "d12_fp16_hq16_b2_padding",
        "q_lens": [2, 4],
        "kv_lens": [128, 256],
        "q_heads": 16,
        "kv_heads": 2,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "identity",
    },
    {
        "name": "d13_fp16_q3_tail_partial",
        "q_lens": [3],
        "kv_lens": [192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "identity",
    },
    {
        "name": "d14_fp16_b2_first_reverse",
        "q_lens": [4, 1],
        "kv_lens": [128, 192],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "first",
        "page_mode": "reverse",
    },
    {
        "name": "d15_bf16_hkv2_rolling",
        "q_lens": [4],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 2,
        "topk": 8,
        "dtype": torch.bfloat16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "d16_fp16_q4_kv384_t16",
        "q_lens": [4],
        "kv_lens": [384],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 16,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "identity",
    },
    {
        "name": "d17_fp16_q8_kv640_tail_t4",
        "q_lens": [8],
        "kv_lens": [640],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
    },
    {
        "name": "d18_fp16_b3_rolling_t8",
        "q_lens": [1, 2, 4],
        "kv_lens": [128, 192, 256],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 8,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "rolling",
        "page_mode": "reverse",
    },
    {
        "name": "d19_fp16_page128_block128_tail",
        "q_lens": [1],
        "kv_lens": [256],
        "q_heads": 8,
        "kv_heads": 1,
        "topk": 4,
        "dtype": torch.float16,
        "causal": True,
        "pattern": "tail",
        "page_mode": "reverse",
        "page_size": 128,
        "sparse_block_size": 128,
    },
]


def test_msa_plan_from_lengths_dense_contract():
    qo_lens = torch.tensor([8, 16], dtype=torch.int64)
    kv_lens = torch.tensor([32, 64], dtype=torch.int64)

    has_mixed, split, batch_size, plan, prefill = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        16,
        num_kv_heads=4,
        causal=True,
    )

    assert has_mixed is False
    assert split == 0
    assert batch_size == 2
    assert prefill is None
    assert plan.mode == "dense"
    assert plan.num_qo_heads == 16
    assert plan.num_kv_heads == 4
    assert plan.kv_page_indptr is None
    assert plan.prefill_plan.total_seqlen_k == 96
    assert torch.equal(
        plan.qo_offset, kv_lens.to(torch.int32) - qo_lens.to(torch.int32)
    )
    assert torch.equal(
        plan.prefill_plan.cu_seqlens_q,
        torch.tensor([0, 8, 24], dtype=torch.int32),
    )


def test_msa_sparse_prefill_mode_is_declared():
    qo_lens = torch.tensor([64], dtype=torch.int32)
    kv_lens = torch.tensor([256], dtype=torch.int32)

    _, _, _, plan, _ = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        16,
        num_kv_heads=4,
        page_size=128,
        kv_block_num=16,
    )

    assert plan.mode == "sparse_prefill"
    assert plan.prefill_plan.total_seqlen_k == 256
    assert torch.equal(
        plan.kv_page_indptr,
        torch.tensor([0, 2], dtype=torch.int32),
    )


def test_msa_plan_from_lengths_keeps_topk_separate_from_selected_block_count():
    _, _, _, plan, _ = msa._msa_plan_from_lengths(
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([4096], dtype=torch.int32),
        8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
    )

    assert plan.kv_block_num == 16
    assert plan.selected_block_count == 18
    assert plan.force_begin_blocks == 1
    assert plan.force_end_blocks == 1
    assert not plan.force_blocks_count_in_topk


def test_msa_plan_requires_device_runtime_metadata():
    plan_info = msa.msa_plan(
        batch_size=1,
        max_seqlen_q=128,
        max_seqlen_k=4096,
        num_qo_heads=8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        sparse_kernel_mode="prefill",
    )
    _, _, _, plan, extra = plan_info
    assert extra is None
    assert plan.is_static
    assert plan.mode == "sparse_prefill"
    assert plan.prefill_plan.max_seqlen_q == 128
    assert plan.prefill_plan.max_seqlen_k == 4096
    assert torch.count_nonzero(plan.qo_lens) == 0

    q = torch.empty((1, 8, 128), dtype=torch.float16)
    k = torch.empty((32, 1, 128, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indexes = torch.zeros((1, 1, 16), dtype=torch.int32)
    with pytest.raises(ValueError, match="static MSA plans require runtime_metadata"):
        msa.sparse_msa(
            q,
            k,
            v,
            plan_info,
            kv_indices=torch.arange(32, dtype=torch.int32),
            kv_block_indexes=block_indexes,
        )


def test_msa_plan_is_the_only_public_capacity_planner():
    assert "msa_plan" in msa.__all__
    assert "msa_plan_static" not in msa.__all__
    assert "msa_plan" in mate.__all__
    assert "msa_plan_static" not in mate.__all__
    assert not hasattr(msa, "msa_plan_static")
    assert not hasattr(mate, "msa_plan_static")


def test_msa_static_runtime_metadata_reaches_fwd_kernel(monkeypatch):
    calls = []

    def fake_msa_fwd(q, k, v, block_indexes, *args, **kwargs):
        calls.append((q, k, v, block_indexes, args, kwargs))
        return torch.empty_like(q), torch.empty((q.shape[0], q.shape[1]))

    monkeypatch.setattr(msa, "_msa_fwd", fake_msa_fwd)
    plan_info = msa.msa_plan(
        batch_size=1,
        max_seqlen_q=128,
        max_seqlen_k=4096,
        num_qo_heads=8,
        num_kv_heads=1,
        page_size=128,
        kv_block_num=16,
        sparse_kernel_mode="prefill",
    )
    runtime = msa.MsaRuntimeMetadata(
        qo_lens=torch.tensor([1], dtype=torch.int32),
        kv_lens=torch.tensor([256], dtype=torch.int32),
        qo_offset=torch.tensor([255], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 256], dtype=torch.int32),
        seqused_k=torch.tensor([256], dtype=torch.int32),
        page_table=torch.cat(
            [
                torch.tensor([[0, 1]], dtype=torch.int32),
                torch.zeros((1, 30), dtype=torch.int32),
            ],
            dim=1,
        ),
    )
    q = torch.empty((1, 8, 128), dtype=torch.float16)
    k = torch.empty((2, 1, 128, 128), dtype=torch.float16)
    v = torch.empty_like(k)
    block_indexes = torch.zeros((1, 1, 16), dtype=torch.int32)
    lse_buffer = torch.empty((1, 8), dtype=torch.float32)
    out, lse = msa.sparse_msa(
        q,
        k,
        v,
        plan_info,
        kv_block_indexes=block_indexes,
        lse=lse_buffer,
        runtime_metadata=runtime,
    )
    assert out.shape == q.shape
    assert lse is None
    assert len(calls) == 1
    # args: cu_seqlens_q, seqused_k, qo_offset, page_table, ...
    assert calls[0][4][0] is runtime.cu_seqlens_q
    assert calls[0][4][1] is runtime.seqused_k
    assert calls[0][4][2] is runtime.qo_offset
    assert calls[0][4][3] is runtime.page_table
    assert calls[0][5]["kv_page_indptr"] is None
    assert calls[0][5]["lse"] is lse_buffer


def test_msa_static_runtime_metadata_reaches_paged_maxscore(monkeypatch):
    calls = []

    def fake_msa_maxscore(q, k, cu_q, cu_k, qo_offset, **kwargs):
        calls.append((q, k, cu_q, cu_k, qo_offset, kwargs))
        return kwargs["max_score"]

    monkeypatch.setattr(msa, "_msa_maxscore", fake_msa_maxscore)
    plan_info = msa.msa_plan(
        batch_size=2,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        total_seqlen_k=8192,
        num_qo_heads=1,
        num_kv_heads=1,
        page_size=128,
        output_maxscore=True,
    )
    _, _, _, plan, _ = plan_info
    assert plan.prefill_plan.total_seqlen_k == 8192

    runtime = msa.MsaRuntimeMetadata(
        qo_lens=torch.ones((2,), dtype=torch.int32),
        kv_lens=torch.tensor([256, 384], dtype=torch.int32),
        qo_offset=torch.tensor([255, 383], dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 256, 640], dtype=torch.int32),
        seqused_k=torch.tensor([256, 384], dtype=torch.int32),
        page_table=torch.zeros((2, 32), dtype=torch.int32),
    )
    q = torch.empty((2, 1, 128), dtype=torch.float16)
    k = torch.empty((64, 128, 1, 128), dtype=torch.float16)
    max_score = torch.empty((2, 1, 128), dtype=torch.float32)
    out, actual = msa.msa(
        q,
        k,
        k,
        plan_info,
        output_o=False,
        output_maxscore=True,
        max_score=max_score,
        runtime_metadata=runtime,
    )

    assert out is None
    assert actual is max_score
    assert len(calls) == 1
    assert calls[0][2] is runtime.cu_seqlens_q
    assert calls[0][3] is runtime.cu_seqlens_k
    assert calls[0][4] is runtime.qo_offset
    assert calls[0][5]["page_table"] is runtime.page_table
    assert calls[0][5]["kv_page_indptr"] is None


def test_sparse_msa_plan_rejects_page_block_size_mismatch():
    with pytest.raises(ValueError, match="page_size == sparse_block_size"):
        msa.sparse_msa_plan(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            sparse_block_size=128,
            kv_block_num=4,
            causal=True,
        )


def test_msa_plan_from_lengths_rejects_page64_paged_maxscore():
    with pytest.raises(ValueError, match="paged MSA maxscore requires page_size"):
        msa._msa_plan_from_lengths(
            torch.tensor([64], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            output_maxscore=True,
            causal=True,
        )


def test_sparse_msa_plan_rejects_unsupported_sparse_block_size():
    with pytest.raises(ValueError, match="sparse_block_size == 128"):
        msa.sparse_msa_plan(
            torch.tensor([256], dtype=torch.int32),
            torch.tensor([256], dtype=torch.int32),
            8,
            num_kv_heads=1,
            page_size=64,
            sparse_block_size=64,
            kv_block_num=4,
            causal=True,
        )


def test_build_page_table_from_flat_kv_indices():
    kv_lens = torch.tensor([256, 384], dtype=torch.int32)
    kv_indices = torch.tensor([3, 5, 7, 11, 13], dtype=torch.int32)

    page_table = msa._build_page_table_from_flat_kv_indices(
        kv_indices,
        kv_lens,
        page_size=128,
    )

    assert torch.equal(
        page_table,
        torch.tensor([[3, 5, 0], [7, 11, 13]], dtype=torch.int32),
    )


def _sparse_topk_select_reference(
    max_score: torch.Tensor,
    *,
    topk: int,
    num_valid_pages: int | None = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    force_blocks_count_in_topk: bool = True,
    query_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    total_q, num_qo_heads, max_k_tiles = max_score.shape
    valid_pages = max_k_tiles if num_valid_pages is None else int(num_valid_pages)
    output_width = (
        topk
        if force_blocks_count_in_topk
        else topk + force_begin_blocks + force_end_blocks
    )
    out = torch.full(
        (total_q, num_qo_heads, output_width),
        -1,
        dtype=torch.int32,
        device=max_score.device,
    )
    for head in range(num_qo_heads):
        for q_abs in range(total_q):
            row_valid_pages = valid_pages
            if query_positions is not None:
                position = int(query_positions[q_abs].item())
                row_valid_pages = (
                    0 if position < 0 else min(valid_pages, position // 128 + 1)
                )
            forced = set(range(min(force_begin_blocks, row_valid_pages)))
            force_end_start = max(0, row_valid_pages - force_end_blocks)
            forced.update(range(force_end_start, row_valid_pages))
            remaining = [idx for idx in range(row_valid_pages) if idx not in forced]
            remaining.sort(
                key=lambda idx: (
                    -float(max_score[q_abs, head, idx].item()),
                    idx,
                )
            )
            if force_blocks_count_in_topk:
                candidates = sorted(forced)
                candidates.extend(remaining[: max(0, topk - len(candidates))])
                candidates = sorted(candidates[:topk])
            else:
                candidates = sorted(forced.union(remaining[:topk]))
            if candidates:
                out[q_abs, head, : len(candidates)] = torch.tensor(
                    candidates,
                    dtype=torch.int32,
                    device=max_score.device,
                )
    return out


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA sparse topk")
@pytest.mark.parametrize("topk", [4, 8, 16])
def test_sparse_topk_select_matches_reference(topk: int):
    device = torch.device("musa")
    torch.manual_seed(401 + topk)
    max_score = torch.randn(
        (3, 17, 128),
        dtype=torch.float32,
        device=device,
    )
    max_score[:, :, 19] += 8.0
    max_score[:, :, 57] += 6.0
    max_score[:, :, 91:] = -float("inf")

    actual = mate.sparse_topk_select(max_score, topk=topk, num_valid_pages=91)
    expected = _sparse_topk_select_reference(
        max_score,
        topk=topk,
        num_valid_pages=91,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA sparse topk")
def test_sparse_topk_select_forced_blocks_and_preallocated_output():
    device = torch.device("musa")
    torch.manual_seed(402)
    max_score = torch.randn(
        (2, 9, 128),
        dtype=torch.float32,
        device=device,
    )
    max_score[:, :, 3] -= 100.0
    max_score[:, :, 125] -= 100.0
    max_score[:, :, 126:] = -float("inf")
    output = torch.empty((2, 9, 8), dtype=torch.int32, device=device)

    actual = mate.sparse_topk_select(
        max_score,
        topk=8,
        num_valid_pages=126,
        output=output,
        force_begin_blocks=4,
        force_end_blocks=3,
    )
    expected = _sparse_topk_select_reference(
        max_score,
        topk=8,
        num_valid_pages=126,
        force_begin_blocks=4,
        force_end_blocks=3,
    )

    assert actual is output
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA sparse topk")
@pytest.mark.parametrize("topk", [4, 8, 16])
@pytest.mark.parametrize("max_k_tiles,num_valid_pages", [(128, 126), (512, 500)])
def test_sparse_topk_select_forced_blocks_outside_topk(
    topk: int, max_k_tiles: int, num_valid_pages: int
):
    device = torch.device("musa")
    torch.manual_seed(403 + max_k_tiles)
    max_score = torch.randn((2, 3, max_k_tiles), dtype=torch.float32, device=device)
    max_score[:, :, 0] = -1000.0
    max_score[:, :, num_valid_pages - 1] = -1000.0
    max_score[:, :, num_valid_pages:] = -float("inf")

    actual = mate.sparse_topk_select(
        max_score,
        topk=topk,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
    )
    expected = _sparse_topk_select_reference(
        max_score,
        topk=topk,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
    )

    assert actual.shape == (2, 3, topk + 2)
    assert torch.equal(actual, expected)
    assert torch.all(actual[..., 0] == 0)
    assert torch.all(actual[..., -1] == num_valid_pages - 1)


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA sparse topk")
def test_sparse_topk_select_trivial_path_clamps_oob_to_tail():
    device = torch.device("musa")
    max_score = torch.zeros((1, 5, 8), dtype=torch.float32, device=device)

    actual = mate.sparse_topk_select(max_score, topk=8, num_valid_pages=5)
    expected = _sparse_topk_select_reference(
        max_score,
        topk=8,
        num_valid_pages=5,
    )

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA sparse topk")
@pytest.mark.parametrize("max_k_tiles,num_valid_pages", [(128, 96), (512, 500)])
def test_sparse_topk_select_uses_query_positions_for_local_block(
    max_k_tiles: int, num_valid_pages: int
):
    device = torch.device("musa")
    torch.manual_seed(404)
    max_score = torch.randn((4, 3, max_k_tiles), dtype=torch.float32, device=device)
    query_positions = torch.tensor(
        [0, 129, 2303, 6144], dtype=torch.int64, device=device
    )

    actual = mate.sparse_topk_select(
        max_score,
        topk=16,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
        query_positions=query_positions,
    )
    expected = _sparse_topk_select_reference(
        max_score,
        topk=16,
        num_valid_pages=num_valid_pages,
        force_begin_blocks=1,
        force_end_blocks=1,
        force_blocks_count_in_topk=False,
        query_positions=query_positions,
    )

    assert torch.equal(actual, expected)
    expected_local = torch.minimum(
        query_positions // 128, torch.tensor(num_valid_pages - 1, device=device)
    )
    for q_abs in range(query_positions.numel()):
        assert int(expected_local[q_abs].item()) in actual[q_abs, 0].tolist()


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA maxscore")
def test_msa_output_maxscore_contiguous_matches_reference():
    device = torch.device("musa")
    torch.manual_seed(301)
    qo_lens = torch.tensor([3, 5], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([140, 65], dtype=torch.int32, device=device)
    num_qo_heads = 4
    num_kv_heads = 2
    head_dim = 128
    total_q = int(qo_lens.sum().item())
    total_k = int(kv_lens.sum().item())
    q = _small_randn(
        (total_q, num_qo_heads, head_dim), device=device, dtype=torch.float16, scale=0.2
    )
    k = _small_randn(
        (total_k, num_kv_heads, head_dim), device=device, dtype=torch.float16, scale=0.2
    )
    v = torch.empty_like(k)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    out, max_score = msa.msa(
        q,
        k,
        v,
        plan_info,
        output_o=False,
        output_maxscore=True,
    )

    assert out is None
    assert max_score is not None
    expected = _dense_msa_output_maxscore_reference(
        q,
        k,
        qo_lens,
        kv_lens,
        num_kv_heads,
        _msa_output_maxscore_k_tiles(int(kv_lens.max().item())),
        causal=True,
    )
    _assert_maxscore_close(max_score, expected)


@pytest.mark.parametrize("mode,q_len", [("prefill", 4), ("decode", 1)])
@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(8, 1), (16, 1), (32, 2)])
@pytest.mark.parametrize("dtype", _FWD_DTYPES)
def test_sparse_modes_route_to_unified_msa_fwd(
    monkeypatch,
    mode: str,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
):
    calls = []

    def fake_msa_fwd(q, k, v, block_indexes, *args, **kwargs):
        calls.append((q, k, v, block_indexes, args, kwargs))
        return (
            torch.empty_like(q),
            torch.empty((q.shape[0], q.shape[1]), dtype=torch.float32),
        )

    monkeypatch.setattr(msa, "_msa_fwd", fake_msa_fwd)
    qo_lens = torch.tensor([q_len], dtype=torch.int32)
    kv_lens = torch.tensor([256], dtype=torch.int32)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_q_heads,
        num_kv_heads=num_kv_heads,
        page_size=128,
        sparse_block_size=128,
        kv_block_num=16,
        sparse_kernel_mode=mode,
        split_prefill_decode=False,
        causal=False,
    )
    q = torch.empty((q_len, num_q_heads, 128), dtype=dtype)
    k = torch.empty((2, num_kv_heads, 128, 128), dtype=dtype)
    v = torch.empty_like(k)
    block_indexes = torch.full((q_len, num_kv_heads, 16), -1, dtype=torch.int32)
    block_indexes[..., :2] = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices = torch.arange(2, dtype=torch.int32)

    if mode == "prefill":
        out, lse = msa.sparse_msa(
            q,
            k,
            v,
            plan_info,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            k_scale=1.25,
            v_scale=1.5,
        )
        assert lse is None
    else:
        out = msa.sparse_decode_atten_func(
            q,
            k,
            v,
            plan_info,
            kv_indices=kv_indices,
            kv_block_indexes=block_indexes,
            k_scale=1.25,
            v_scale=1.5,
        )

    assert out.shape == q.shape
    assert len(calls) == 1
    assert calls[0][3] is block_indexes
    assert calls[0][5]["k_scale"] == 1.25
    assert calls[0][5]["v_scale"] == 1.5


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA maxscore")
def test_msa_output_maxscore_paged_preallocated_matches_reference():
    device = torch.device("musa")
    torch.manual_seed(302)
    page_size = 128
    qo_lens = torch.tensor([2, 4], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([130, 70], dtype=torch.int32, device=device)
    num_qo_heads = 4
    num_kv_heads = 2
    head_dim = 128
    total_q = int(qo_lens.sum().item())
    page_table, kv_indices = _make_page_table_and_kv_indices(
        kv_lens,
        page_size=page_size,
        mode="reverse",
    )
    num_pages = int(kv_indices.numel())
    q = _small_randn(
        (total_q, num_qo_heads, head_dim), device=device, dtype=torch.float16, scale=0.2
    )
    k_pages = _small_randn(
        (num_pages, num_kv_heads, page_size, head_dim),
        device=device,
        dtype=torch.float16,
        scale=0.2,
    )
    v_pages = torch.empty_like(k_pages)
    max_k_tiles = _msa_output_maxscore_k_tiles(int(kv_lens.max().item()))
    max_score_prealloc = torch.empty(
        (total_q, num_qo_heads, max_k_tiles),
        dtype=torch.float32,
        device=device,
    )
    plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    out, max_score = msa.msa(
        q,
        k_pages,
        v_pages,
        plan_info,
        kv_indices=kv_indices,
        output_o=False,
        max_score=max_score_prealloc,
    )

    assert out is None
    assert max_score is max_score_prealloc
    expected = _dense_msa_output_maxscore_reference(
        q,
        k_pages,
        qo_lens,
        kv_lens,
        num_kv_heads,
        max_k_tiles,
        causal=True,
        page_table=page_table,
        page_size=page_size,
    )
    _assert_maxscore_close(max_score, expected)


@pytest.mark.skipif(not _HAS_MUSA, reason="MUSA is required for MSA bf16 maxscore")
def test_msa_output_maxscore_bf16_contiguous_and_paged_match_reference():
    device = torch.device("musa")
    torch.manual_seed(304)
    page_size = 128
    qo_lens = torch.tensor([33, 64], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([257, 130], dtype=torch.int32, device=device)
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    total_q = int(qo_lens.sum().item())
    total_k = int(kv_lens.sum().item())
    dtype = torch.bfloat16
    max_k_tiles = _msa_output_maxscore_k_tiles(int(kv_lens.max().item()))

    q = _small_randn(
        (total_q, num_qo_heads, head_dim), device=device, dtype=dtype, scale=0.2
    )
    k = _small_randn(
        (total_k, num_kv_heads, head_dim), device=device, dtype=dtype, scale=0.2
    )
    v = torch.empty_like(k)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    out, max_score = msa.msa(
        q,
        k,
        v,
        plan_info,
        output_o=False,
        output_maxscore=True,
    )

    assert out is None
    assert max_score is not None
    expected = _dense_msa_output_maxscore_reference(
        q,
        k,
        qo_lens,
        kv_lens,
        num_kv_heads,
        max_k_tiles,
        causal=True,
    )
    _assert_maxscore_close(max_score, expected)

    page_table, kv_indices = _make_page_table_and_kv_indices(
        kv_lens,
        page_size=page_size,
        mode="reverse",
    )
    num_pages = int(kv_indices.numel())
    k_pages = _small_randn(
        (num_pages, num_kv_heads, page_size, head_dim),
        device=device,
        dtype=dtype,
        scale=0.2,
    )
    v_pages = torch.empty_like(k_pages)
    paged_plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    paged_out, paged_max_score = msa.msa(
        q,
        k_pages,
        v_pages,
        paged_plan_info,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
    )

    assert paged_out is None
    assert paged_max_score is not None
    paged_expected = _dense_msa_output_maxscore_reference(
        q,
        k_pages,
        qo_lens,
        kv_lens,
        num_kv_heads,
        max_k_tiles,
        causal=True,
        page_table=page_table,
        page_size=page_size,
    )
    _assert_maxscore_close(paged_max_score, paged_expected)


@pytest.mark.skipif(
    not (_HAS_MUSA and _HAS_FP8_E4M3),
    reason="MUSA and torch.float8_e4m3fn are required for MSA fp8 maxscore",
)
def test_msa_output_maxscore_fp8_contiguous_and_paged_match_reference():
    device = torch.device("musa")
    torch.manual_seed(303)
    page_size = 128
    qo_lens = torch.tensor([33, 64], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([257, 130], dtype=torch.int32, device=device)
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    total_q = int(qo_lens.sum().item())
    total_k = int(kv_lens.sum().item())
    dtype = torch.float8_e4m3fn
    max_k_tiles = _msa_output_maxscore_k_tiles(int(kv_lens.max().item()))

    q = _small_randn(
        (total_q, num_qo_heads, head_dim), device=device, dtype=dtype, scale=0.2
    )
    k = _small_randn(
        (total_k, num_kv_heads, head_dim), device=device, dtype=dtype, scale=0.2
    )
    v = torch.empty_like(k)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    out, max_score = msa.msa(
        q,
        k,
        v,
        plan_info,
        output_o=False,
        output_maxscore=True,
    )

    assert out is None
    assert max_score is not None
    expected = _dense_msa_output_maxscore_reference(
        q,
        k,
        qo_lens,
        kv_lens,
        num_kv_heads,
        max_k_tiles,
        causal=True,
    )
    _assert_maxscore_close(max_score, expected)

    page_table, kv_indices = _make_page_table_and_kv_indices(
        kv_lens,
        page_size=page_size,
        mode="reverse",
    )
    num_pages = int(kv_indices.numel())
    k_pages = _small_randn(
        (num_pages, num_kv_heads, page_size, head_dim),
        device=device,
        dtype=dtype,
        scale=0.2,
    )
    v_pages = torch.empty_like(k_pages)
    paged_plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    paged_out, paged_max_score = msa.msa(
        q,
        k_pages,
        v_pages,
        paged_plan_info,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
    )

    assert paged_out is None
    assert paged_max_score is not None
    paged_expected = _dense_msa_output_maxscore_reference(
        q,
        k_pages,
        qo_lens,
        kv_lens,
        num_kv_heads,
        max_k_tiles,
        causal=True,
        page_table=page_table,
        page_size=page_size,
    )
    _assert_maxscore_close(paged_max_score, paged_expected)


@pytest.mark.skipif(
    not (_HAS_MUSA and _HAS_FP8_E4M3),
    reason="MUSA and torch.float8_e4m3fn are required for MSA fp8 maxscore",
)
def test_msa_output_maxscore_fp8_decode_parallel_k_matches_reference():
    device = torch.device("musa")
    torch.manual_seed(305)
    page_size = 128
    qo_lens = torch.tensor([1, 1], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([385, 130], dtype=torch.int32, device=device)
    num_qo_heads = 16
    num_kv_heads = 1
    head_dim = 128
    dtype = torch.float8_e4m3fn
    total_q = int(qo_lens.sum().item())
    max_k_tiles = _msa_output_maxscore_k_tiles(int(kv_lens.max().item()))

    q = _small_randn(
        (total_q, num_qo_heads, head_dim), device=device, dtype=dtype, scale=0.2
    )
    page_table, kv_indices = _make_page_table_and_kv_indices(
        kv_lens,
        page_size=page_size,
        mode="reverse",
    )
    num_pages = int(kv_indices.numel())
    k_pages = _small_randn(
        (num_pages, num_kv_heads, page_size, head_dim),
        device=device,
        dtype=dtype,
        scale=0.2,
    )
    v_pages = torch.empty_like(k_pages)
    plan_info = msa._msa_plan_from_lengths(
        qo_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        causal=True,
        output_maxscore=True,
        split_prefill_decode=False,
    )

    out, max_score = msa.msa(
        q,
        k_pages,
        v_pages,
        plan_info,
        kv_indices=kv_indices,
        output_o=False,
        output_maxscore=True,
    )

    assert out is None
    assert max_score is not None
    expected = _dense_msa_output_maxscore_reference(
        q,
        k_pages,
        qo_lens,
        kv_lens,
        num_kv_heads,
        max_k_tiles,
        causal=True,
        page_table=page_table,
        page_size=page_size,
    )
    _assert_maxscore_close(max_score, expected)
