from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Mapping, Optional, cast

import torch

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from . import env as jit_env
from .core import JitSpec, gen_jit_spec
from .utils import maybe_contiguous


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


def _maxscore_k_tiles(max_seqlen_k: int) -> int:
    kv_tiles = (int(max_seqlen_k) + 127) // 128
    return ((kv_tiles + 127) // 128) * 128


def _msa_maxscore_encode(config: Mapping[str, object]) -> str:
    return (
        f"msa_maxscore_dtype_{config['dtype_name']}"
        f"_paged_{int(bool(config['is_paged_kv']))}"
        f"_causal_{int(bool(config['causal']))}"
        f"_hd_{config['head_dim']}"
        f"_hr_{config['head_ratio']}"
        f"_tq_{config['tile_q']}"
        f"_qstages_{config['q_stages']}"
        f"_kstages_{config['k_stages']}"
        f"_pk_{int(bool(config['parallel_k_tiles']))}"
    )


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
    if parallel_k_tiles and 16 % int(head_ratio) == 0:
        return 16
    tile_q = max(64, int(head_ratio))
    while tile_q % int(head_ratio) != 0 or tile_q % 4 != 0:
        tile_q += 4
    return tile_q


def _maxscore_stage_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    stage = int(value)
    if stage <= 0:
        raise ValueError(f"{name} must be positive, got {stage}")
    return stage


def make_msa_maxscore_config(
    dtype: torch.dtype,
    *,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    q_stages: int | None = None,
    k_stages: int | None = None,
    parallel_k_tiles: bool = False,
    tile_q: int | None = None,
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
    q_stages = (
        _maxscore_stage_env("MATE_MSA_MAXSCORE_Q_STAGES", 1)
        if q_stages is None
        else int(q_stages)
    )
    k_stages = (
        _maxscore_stage_env("MATE_MSA_MAXSCORE_K_STAGES", 1)
        if k_stages is None
        else int(k_stages)
    )
    if q_stages <= 0 or k_stages <= 0:
        raise ValueError(
            "msa maxscore q_stages and k_stages must be positive, "
            f"got {q_stages}, {k_stages}"
        )
    selected_tile_q = (
        _maxscore_tile_q(int(head_ratio), bool(parallel_k_tiles), dtype=dtype)
        if tile_q is None
        else int(tile_q)
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
        "q_stages": int(q_stages),
        "k_stages": int(k_stages),
        "parallel_k_tiles": bool(parallel_k_tiles),
    }


def _render_msa_maxscore_kernel_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = _msa_maxscore_encode(config)
    return MSA_MAXSCORE_KERN_TEMPLATE.render(render_config)


def gen_msa_maxscore_spec(
    dtype: torch.dtype,
    *,
    is_paged_kv: bool,
    causal: bool,
    head_dim: int,
    head_ratio: int,
    q_stages: int | None = None,
    k_stages: int | None = None,
    parallel_k_tiles: bool = False,
    tile_q: int | None = None,
) -> JitSpec:
    config = make_msa_maxscore_config(
        dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        q_stages=q_stages,
        k_stages=k_stages,
        parallel_k_tiles=parallel_k_tiles,
        tile_q=tile_q,
    )
    dispatch_name = _msa_maxscore_encode(config)
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
    q_stages: int,
    k_stages: int,
    parallel_k_tiles: bool = False,
    tile_q: int | None = None,
):
    return gen_msa_maxscore_spec(
        dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        q_stages=q_stages,
        k_stages=k_stages,
        parallel_k_tiles=parallel_k_tiles,
        tile_q=tile_q,
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

    # Decode has too few Q tiles to occupy all MPs, so it always partitions K.
    # The MSA selector uses one proxy-Q head per KV head (head_ratio == 1).
    # For causal prefill at 4K and above, Q tiles plus parallel K partitions
    # provide enough independent work to keep all MPs occupied.  TileQ is then
    # selected separately for short and long contexts above.
    parallel_k_tiles = int(max_seqlen_q) == 1 or (
        causal and head_ratio == 1 and int(max_seqlen_q) >= 4096
    )
    tile_q = _maxscore_tile_q(
        int(head_ratio),
        bool(parallel_k_tiles),
        dtype=q.dtype,
        max_seqlen_q=int(max_seqlen_q),
    )
    config = make_msa_maxscore_config(
        q.dtype,
        is_paged_kv=is_paged_kv,
        causal=causal,
        head_dim=head_dim,
        head_ratio=head_ratio,
        parallel_k_tiles=parallel_k_tiles,
        tile_q=tile_q,
    )
    dispatch_name = _msa_maxscore_encode(config)
    kernel = get_msa_maxscore_module(
        q.dtype,
        is_paged_kv,
        bool(causal),
        head_dim,
        head_ratio,
        cast(int, config["q_stages"]),
        cast(int, config["k_stages"]),
        bool(config["parallel_k_tiles"]),
        cast(int, config["tile_q"]),
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


def gen_msa_ops_spec() -> JitSpec:
    from .msa_fwd import gen_msa_fwd_spec

    return gen_msa_fwd_spec(causal=True)


def gen_msa_ops_aot() -> list[JitSpec]:
    from .msa_fwd import gen_msa_fwd_spec

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
    from .msa_fwd import get_msa_fwd_module

    if _FP8_E4M3_DTYPE is None:
        raise RuntimeError("torch.float8_e4m3fn is unavailable")
    return get_msa_fwd_module(_FP8_E4M3_DTYPE, causal=True)
