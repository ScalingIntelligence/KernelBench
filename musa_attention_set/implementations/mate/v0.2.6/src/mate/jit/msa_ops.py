from __future__ import annotations

import functools
import math
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional, cast

import torch

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import TVM_HEADER, maybe_contiguous


CXX_FLAGS = [
    "-O3",
    "-Wno-switch-bool",
]

CUDA_FLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

INCLUDE_PATHS = [
    jit_env.MATE_INCLUDE_DIR,
    jit_env.MATE_CSRC_DIR,
    jit_env.MUTLASS_INCLUDE_DIR,
]


@functools.cache
def _get_msa_template_env(template_dir: str) -> Environment:
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def get_msa_template(template_name: str):
    template_dir = (jit_env.MATE_TEMPLATE_DIR / "attention" / "msa").as_posix()
    return _get_msa_template_env(template_dir).get_template(template_name)


MSA_MAXSCORE_KERN_TEMPLATE = get_msa_template("maxscore_kern.j2")
MSA_SPARSE_TOPK_SELECT_TEMPLATE = get_msa_template("sparse_topk_select_kern.j2")
MSA_FWD_KERN_TEMPLATE = get_msa_template("fwd_kern.j2")
_FP8_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)

MSA_DTYPE_MUTLASS = {
    torch.float16: "mutlass::half_t",
    torch.bfloat16: "mutlass::bfloat16_t",
}
if _FP8_E4M3_DTYPE is not None:
    MSA_DTYPE_MUTLASS[_FP8_E4M3_DTYPE] = "mutlass::float_e4m3_t"

MSA_DTYPE_FFI = {
    torch.float16: "dl_float16",
    torch.bfloat16: "dl_bfloat16",
}
if _FP8_E4M3_DTYPE is not None:
    MSA_DTYPE_FFI[_FP8_E4M3_DTYPE] = "dl_float8_e4m3fn"

MSA_DTYPE_NAMES = {
    torch.float16: "f16",
    torch.bfloat16: "bf16",
}
if _FP8_E4M3_DTYPE is not None:
    MSA_DTYPE_NAMES[_FP8_E4M3_DTYPE] = "fp8e4m3"

MSA_FWD_DTYPE_CONFIG: dict[torch.dtype, dict[str, str]] = {
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
    MSA_FWD_DTYPE_CONFIG[_FP8_E4M3_DTYPE] = {
        "name": "fp8e4m3",
        "element": "mutlass::float_e4m3_t",
        "element_dtype": "dl_float8_e4m3fn",
    }


def _maxscore_k_tiles(max_seqlen_k: int) -> int:
    kv_tiles = (int(max_seqlen_k) + 127) // 128
    return ((kv_tiles + 127) // 128) * 128


@lru_cache(maxsize=256)
def _msa_maxscore_encode(
    dtype_name: str,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    tile_q: int,
    parallel_k_tiles: bool,
) -> str:
    return (
        f"msa_maxscore_dtype_{dtype_name}"
        f"_paged_{int(is_paged_kv)}"
        f"_causal_{int(causal)}"
        f"_hd_{head_dim}"
        f"_hr_{head_ratio}"
        f"_tq_{tile_q}"
        f"_pk_{int(parallel_k_tiles)}"
    )


def _msa_maxscore_encode_config(config: Mapping[str, object]) -> str:
    return _msa_maxscore_encode(
        str(config["dtype_name"]),
        bool(config["is_paged_kv"]),
        bool(config["causal"]),
        cast(int, config["head_dim"]),
        cast(int, config["head_ratio"]),
        cast(int, config["tile_q"]),
        bool(config["parallel_k_tiles"]),
    )


@lru_cache(maxsize=256)
def _maxscore_tile_q(
    head_ratio: int,
    parallel_k_tiles: bool,
    *,
    dtype: torch.dtype | None = None,
    max_seqlen_q: int | None = None,
) -> int:
    # A larger Q tile amortizes the K TME transaction and pipeline barriers
    # over more rows.  It only pays off once a prefill has enough Q tiles to
    # keep one 20-warp CTA per MP resident; short prefill and decode retain
    # the 8-warp TQ16 variant.  The larger variant is currently validated for
    # BF16 only; FP8's TCE path regresses at the same shape.
    if (
        dtype == torch.bfloat16
        and parallel_k_tiles
        and int(head_ratio) == 1
        and max_seqlen_q is not None
        and int(max_seqlen_q) >= 64 * 1024
    ):
        return 128
    if (
        parallel_k_tiles
        and max_seqlen_q is not None
        and (int(max_seqlen_q) == 1 or dtype == torch.bfloat16)
        and 16 % int(head_ratio) == 0
    ):
        return 16
    tile_q = max(64, int(head_ratio))
    while tile_q % int(head_ratio) != 0 or tile_q % 4 != 0:
        tile_q += 4
    return tile_q


@lru_cache(maxsize=256)
def make_msa_maxscore_config(
    dtype: torch.dtype,
    *,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    max_seqlen_q: int,
) -> dict[str, object]:
    if dtype not in MSA_DTYPE_MUTLASS:
        raise TypeError(
            "msa maxscore only supports torch.float16, torch.bfloat16, "
            "and torch.float8_e4m3fn."
        )
    if int(head_dim) <= 0 or int(head_dim) % 8 != 0:
        raise ValueError(
            f"msa maxscore expects head_dim divisible by 8, got {head_dim}"
        )
    if int(head_ratio) <= 0:
        raise ValueError(f"msa maxscore expects positive head_ratio, got {head_ratio}")
    # The MSA selector uses one proxy-Q head per KV head.  Split K whenever that
    # contract holds so short-Q and tail waves expose enough independent work.
    # The scheduler chooses the partition count from the actual Q-work capacity.
    parallel_k_tiles = int(head_ratio) == 1
    selected_tile_q = _maxscore_tile_q(
        int(head_ratio),
        bool(parallel_k_tiles),
        dtype=dtype,
        max_seqlen_q=int(max_seqlen_q),
    )
    if (
        selected_tile_q <= 0
        or selected_tile_q % 4 != 0
        or selected_tile_q % int(head_ratio) != 0
    ):
        raise ValueError(
            "msa maxscore tile_q must be a positive multiple of 4 and "
            f"head_ratio, got tile_q={selected_tile_q}, head_ratio={head_ratio}"
        )
    return {
        "element": MSA_DTYPE_MUTLASS[dtype],
        "element_dtype": MSA_DTYPE_FFI[dtype],
        "dtype_name": MSA_DTYPE_NAMES[dtype],
        "is_paged_kv": bool(is_paged_kv),
        "causal": bool(causal),
        "head_dim": int(head_dim),
        "head_ratio": int(head_ratio),
        "tile_q": selected_tile_q,
        "parallel_k_tiles": bool(parallel_k_tiles),
    }


def _render_msa_maxscore_kernel_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = _msa_maxscore_encode_config(config)
    return MSA_MAXSCORE_KERN_TEMPLATE.render(render_config)


def gen_msa_maxscore_spec(
    dtype: torch.dtype,
    *,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    max_seqlen_q: int,
) -> JitSpec:
    config = make_msa_maxscore_config(
        dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        max_seqlen_q=max_seqlen_q,
    )
    dispatch_name = _msa_maxscore_encode_config(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / f"{dispatch_name}.mu")
    return gen_jit_spec(
        dispatch_name,
        [source_file],
        generated_sources={source_file: _render_msa_maxscore_kernel_source(config)},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def get_msa_maxscore_module(
    dtype: torch.dtype,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    max_seqlen_q: int,
):
    return gen_msa_maxscore_spec(
        dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        max_seqlen_q=max_seqlen_q,
    ).build_and_load()


def _msa_maxscore(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    qo_offset: torch.Tensor,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    page_table: Optional[torch.Tensor] = None,
    kv_page_indptr: Optional[torch.Tensor] = None,
    max_score: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    cu_seqlens_q = maybe_contiguous(cu_seqlens_q)
    cu_seqlens_k = maybe_contiguous(cu_seqlens_k)
    qo_offset = maybe_contiguous(qo_offset)
    page_table = maybe_contiguous(page_table)
    kv_page_indptr = maybe_contiguous(kv_page_indptr)
    if q.ndim != 3:
        raise ValueError(f"msa maxscore expects packed q with rank 3, got {q.shape}")
    if q.dtype != k.dtype:
        raise TypeError(f"q and k must share dtype, got {q.dtype} and {k.dtype}")
    if q.dtype not in MSA_DTYPE_MUTLASS:
        raise TypeError(f"unsupported MSA maxscore dtype: {q.dtype}")
    is_paged_kv = page_table is not None
    if kv_page_indptr is not None:
        if page_table is None:
            raise ValueError("kv_page_indptr requires flat page indices")
        if page_table.ndim != 1:
            raise ValueError(
                "flat paged maxscore expects rank-1 page indices when "
                "kv_page_indptr is provided"
            )
        if kv_page_indptr.ndim != 1:
            raise ValueError("kv_page_indptr must be rank 1")
    expected_k_rank = 4 if is_paged_kv else 3
    if k.ndim != expected_k_rank:
        raise ValueError(
            f"msa maxscore expected rank-{expected_k_rank} k, got shape {k.shape}"
        )
    total_q, num_qo_heads, head_dim = (
        int(q.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
    )
    num_kv_heads = int(k.shape[2] if is_paged_kv else k.shape[1])
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "num_qo_heads must be divisible by num_kv_heads, got "
            f"{num_qo_heads} and {num_kv_heads}"
        )
    head_ratio = num_qo_heads // num_kv_heads
    if int(k.shape[-1]) != head_dim:
        raise ValueError(f"q/k head_dim mismatch: {head_dim} vs {int(k.shape[-1])}")

    valid_k_tiles = (int(max_seqlen_k) + 127) // 128
    max_k_tiles = _maxscore_k_tiles(max_seqlen_k)
    if max_score is None:
        max_score = torch.full(
            (total_q, num_qo_heads, max_k_tiles),
            -torch.inf,
            dtype=torch.float32,
            device=q.device,
        )
    else:
        max_score = maybe_contiguous(max_score)
        expected_shape = (total_q, num_qo_heads, max_k_tiles)
        if tuple(max_score.shape) != expected_shape:
            raise ValueError(
                f"max_score shape must be {expected_shape}, got {tuple(max_score.shape)}"
            )
        if max_score.dtype != torch.float32:
            raise TypeError(f"max_score must be torch.float32, got {max_score.dtype}")
        # The causal max-score kernel skips K tiles that are wholly in the
        # future of a Q tile.  Keep the dense output contract by initializing
        # those entries (and any padded tail) to -inf before the kernel writes
        # the visible tiles.
        if causal or valid_k_tiles < max_k_tiles:
            max_score.fill_(-torch.inf)

    if total_q == 0 or valid_k_tiles == 0:
        return max_score

    config = make_msa_maxscore_config(
        q.dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        max_seqlen_q=int(max_seqlen_q),
    )
    dispatch_name = _msa_maxscore_encode_config(config)
    kernel = get_msa_maxscore_module(
        q.dtype,
        is_paged_kv,
        bool(causal),
        head_dim,
        head_ratio,
        int(max_seqlen_q),
    ).get_function(dispatch_name)
    kernel(
        q,
        k,
        cu_seqlens_q,
        cu_seqlens_k,
        qo_offset,
        page_table,
        kv_page_indptr,
        max_score,
        int(max_seqlen_q),
        int(max_seqlen_k),
        bool(causal),
    )
    return max_score


def gen_msa_sparse_topk_select_spec() -> JitSpec:
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / "sparse_topk_select.mu")
    return gen_jit_spec(
        "msa_sparse_topk_select",
        [source_file],
        generated_sources={source_file: MSA_SPARSE_TOPK_SELECT_TEMPLATE.render({})},
        extra_cflags=list(CXX_FLAGS),
        extra_cuda_cflags=list(CUDA_FLAGS),
        extra_include_paths=list(INCLUDE_PATHS),
    )


@functools.cache
def get_msa_sparse_topk_select_module():
    return gen_msa_sparse_topk_select_spec().build_and_load()


def _msa_sparse_topk_select(
    max_score: torch.Tensor,
    output_indices: torch.Tensor,
    *,
    topk: int,
    num_valid_pages: int,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
    force_blocks_count_in_topk: bool = True,
    query_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    max_score = maybe_contiguous(max_score)
    output_indices = maybe_contiguous(output_indices)
    get_msa_sparse_topk_select_module().get_function("sparse_topk_select")(
        max_score,
        output_indices,
        int(topk),
        int(num_valid_pages),
        int(force_begin_blocks),
        int(force_end_blocks),
        bool(force_blocks_count_in_topk),
        query_positions,
    )
    return output_indices


@lru_cache(maxsize=256)
def _resolve_fwd_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is None:
        if _FP8_E4M3_DTYPE is None:
            raise RuntimeError("torch.float8_e4m3fn is unavailable")
        return _FP8_E4M3_DTYPE
    if dtype not in MSA_FWD_DTYPE_CONFIG:
        raise TypeError(f"unsupported MSA forward dtype: {dtype}")
    return dtype


@lru_cache(maxsize=256)
def _msa_fwd_encode(*, dtype: torch.dtype, causal: bool) -> str:
    return f"msa_fwd_{MSA_FWD_DTYPE_CONFIG[dtype]['name']}_causal_{int(causal)}"


@lru_cache(maxsize=256)
def make_msa_fwd_config(
    *,
    causal: bool,
    dtype: Optional[torch.dtype] = None,
) -> dict[str, object]:
    dtype = _resolve_fwd_dtype(dtype)
    return {
        **MSA_FWD_DTYPE_CONFIG[dtype],
        "causal": bool(causal),
        "head_ratio": 16,
        "head_dim": 128,
        "tile_kv": 128,
        "topk": 16,
    }


def _render_msa_fwd_kernel_source(config: dict[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = (
        f"msa_fwd_{config['name']}_causal_{int(bool(config['causal']))}"
    )
    return TVM_HEADER + MSA_FWD_KERN_TEMPLATE.render(render_config)


def gen_msa_fwd_spec(
    *,
    causal: bool,
    dtype: Optional[torch.dtype] = None,
) -> JitSpec:
    dtype = _resolve_fwd_dtype(dtype)
    config = make_msa_fwd_config(causal=causal, dtype=dtype)
    dispatch_name = _msa_fwd_encode(dtype=dtype, causal=causal)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / "msa" / f"{dispatch_name}.mu")
    return gen_jit_spec(
        dispatch_name,
        [source_file],
        generated_sources={source_file: _render_msa_fwd_kernel_source(config)},
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

    if q.dtype not in MSA_FWD_DTYPE_CONFIG:
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
    dispatch_name = _msa_fwd_encode(dtype=q.dtype, causal=bool(config["causal"]))
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


def gen_msa_ops_spec() -> JitSpec:
    return gen_msa_fwd_spec(causal=True)


def gen_msa_ops_aot() -> list[JitSpec]:
    dtypes = [torch.float16, torch.bfloat16]
    if _FP8_E4M3_DTYPE is not None:
        dtypes.append(_FP8_E4M3_DTYPE)
    specs = [
        gen_msa_fwd_spec(dtype=dtype, causal=causal)
        for dtype in dtypes
        for causal in (False, True)
    ]
    specs.append(gen_msa_sparse_topk_select_spec())
    return specs


@functools.cache
def get_msa_ops_module():
    if _FP8_E4M3_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    return get_msa_fwd_module(_FP8_E4M3_DTYPE, causal=True)
