from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Optional

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .msa_ops import CXX_FLAGS, CUDA_FLAGS, INCLUDE_PATHS, get_msa_template
from .utils import TVM_HEADER, maybe_contiguous


_FP8_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)
_TEMPLATE = get_msa_template("fwd_kern.j2")
_DTYPE_CONFIG: dict[torch.dtype, dict[str, str]] = {
    torch.float16: {
        "name": "f16",
        "element": "mutlass::half_t",
        "element_dtype": "dl_float16",
    },
    torch.bfloat16: {
        "name": "bf16",
        "element": "mutlass::bfloat16_t",
        "element_dtype": "dl_bfloat16",
    },
}
if _FP8_E4M3_DTYPE is not None:
    _DTYPE_CONFIG[_FP8_E4M3_DTYPE] = {
        "name": "fp8e4m3",
        "element": "mutlass::float_e4m3_t",
        "element_dtype": "dl_float8_e4m3fn",
    }


def _resolve_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        if _FP8_E4M3_DTYPE is None:
            raise RuntimeError("torch.float8_e4m3fn is unavailable")
        return _FP8_E4M3_DTYPE
    if dtype not in _DTYPE_CONFIG:
        raise TypeError(f"unsupported MSA forward dtype: {dtype}")
    return dtype


def _encode(*, dtype: torch.dtype, causal: bool) -> str:
    return f"msa_fwd_{_DTYPE_CONFIG[dtype]['name']}_causal_{int(causal)}"


def make_msa_fwd_config(
    *,
    causal: bool,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, object]:
    dtype = _resolve_dtype(dtype)
    return {
        **_DTYPE_CONFIG[dtype],
        "causal": bool(causal),
        "head_ratio": 16,
        "head_dim": 128,
        "tile_kv": 128,
        "topk": 16,
    }


def _render(config: dict[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = (
        f"msa_fwd_{config['name']}_causal_{int(bool(config['causal']))}"
    )
    return TVM_HEADER + _TEMPLATE.render(render_config)


def gen_msa_fwd_spec(
    *,
    causal: bool,
    dtype: Optional[torch.dtype] = None,
) -> JitSpec:
    dtype = _resolve_dtype(dtype)
    config = make_msa_fwd_config(causal=causal, dtype=dtype)
    dispatch_name = _encode(dtype=dtype, causal=causal)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / f"{dispatch_name}.mu")
    return gen_jit_spec(
        dispatch_name,
        [source_file],
        generated_sources={source_file: _render(config)},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def get_msa_fwd_module(dtype: torch.dtype, causal: bool):
    return gen_msa_fwd_spec(dtype=dtype, causal=causal).build_and_load()


def _require_int32_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    value = maybe_contiguous(value)
    if value.ndim != 1 or value.dtype != torch.int32:
        raise TypeError(f"{name} must be a contiguous rank-1 int32 tensor")
    return value


def _msa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_block_indexes: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    qo_offset: torch.Tensor,
    page_table: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    kv_page_indptr: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    v = maybe_contiguous(v)
    kv_block_indexes = maybe_contiguous(kv_block_indexes)
    page_table = maybe_contiguous(page_table)
    cu_seqlens_q = _require_int32_vector("cu_seqlens_q", cu_seqlens_q)
    seqused_k = _require_int32_vector("seqused_k", seqused_k)
    qo_offset = _require_int32_vector("qo_offset", qo_offset)
    if kv_page_indptr is not None:
        kv_page_indptr = _require_int32_vector("kv_page_indptr", kv_page_indptr)

    if q.dtype not in _DTYPE_CONFIG:
        raise TypeError("MSA forward supports float8_e4m3fn, float16, and bfloat16")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise TypeError("q, k, and v must share the same dtype")
    if q.ndim != 3 or int(q.shape[-1]) != 128:
        raise ValueError(f"q must have shape [total_q, Hq, 128], got {tuple(q.shape)}")
    if k.ndim != 4 or v.ndim != 4 or tuple(k.shape) != tuple(v.shape):
        raise ValueError("k and v must share shape [num_pages, 128, Hkv, 128]")
    if int(k.shape[1]) != 128 or int(k.shape[-1]) != 128:
        raise ValueError(f"paged k/v must use page_size=D=128, got {tuple(k.shape)}")
    k_scale = float(k_scale)
    v_scale = float(v_scale)
    if not math.isfinite(k_scale) or k_scale <= 0.0:
        raise ValueError(f"k_scale must be finite and positive, got {k_scale}")
    if not math.isfinite(v_scale) or v_scale <= 0.0:
        raise ValueError(f"v_scale must be finite and positive, got {v_scale}")

    total_q = int(q.shape[0])
    num_q_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[2])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads != 0:
        raise ValueError("MSA SQMMA forward requires Hq to be divisible by Hkv")
    if num_q_heads // num_kv_heads not in (8, 16):
        raise ValueError("MSA SQMMA forward requires local Hq/Hkv ratio 8 or 16")
    expected_blocks_shape = (total_q, num_kv_heads, 16)
    if tuple(kv_block_indexes.shape) != expected_blocks_shape:
        raise ValueError(
            f"kv_block_indexes must have shape {expected_blocks_shape}, "
            f"got {tuple(kv_block_indexes.shape)}"
        )
    if kv_block_indexes.dtype != torch.int32:
        raise TypeError("kv_block_indexes must use torch.int32")

    batch_size = int(cu_seqlens_q.numel() - 1)
    if tuple(seqused_k.shape) != (batch_size,):
        raise ValueError("seqused_k must contain one value per batch")
    if tuple(qo_offset.shape) != (batch_size,):
        raise ValueError("qo_offset must contain one value per batch")
    if kv_page_indptr is not None:
        if page_table.ndim != 1 or int(kv_page_indptr.numel()) != batch_size + 1:
            raise ValueError(
                "flat page_table requires kv_page_indptr with batch_size + 1 entries"
            )
    elif page_table.ndim != 2 or int(page_table.shape[0]) != batch_size:
        raise ValueError(
            "page_table must be [batch, max_pages] when kv_page_indptr is absent"
        )
    elif int(page_table.shape[1]) != (int(max_seqlen_k) + 127) // 128:
        raise ValueError(
            "page_table width must equal ceil(max_seqlen_k / 128) for the "
            "MSA forward TME descriptor"
        )
    if page_table.dtype != torch.int32:
        raise TypeError("page_table must use torch.int32")

    if out is None:
        out = torch.empty_like(q)
    elif tuple(out.shape) != tuple(q.shape) or out.dtype != q.dtype:
        raise ValueError("out must have the same shape and dtype as q")
    if lse is None:
        lse = torch.empty((total_q, num_q_heads), dtype=torch.float32, device=q.device)
    elif tuple(lse.shape) != (total_q, num_q_heads) or lse.dtype != torch.float32:
        raise ValueError("lse must have shape [total_q, Hq] and dtype float32")

    config = make_msa_fwd_config(causal=causal, dtype=q.dtype)
    dispatch_name = _encode(dtype=q.dtype, causal=bool(config["causal"]))
    kernel = get_msa_fwd_module(q.dtype, causal).get_function(dispatch_name)
    kernel(
        q,
        k,
        v,
        kv_block_indexes,
        cu_seqlens_q,
        seqused_k,
        qo_offset,
        page_table,
        kv_page_indptr,
        int(max_seqlen_q),
        int(max_seqlen_k),
        float(softmax_scale if softmax_scale is not None else 128**-0.5),
        k_scale,
        v_scale,
        bool(causal),
        out,
        lse,
    )
    return out, lse
