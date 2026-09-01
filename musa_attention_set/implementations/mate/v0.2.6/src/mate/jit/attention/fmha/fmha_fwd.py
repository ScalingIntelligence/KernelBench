import functools
from pathlib import Path
import torch
from typing import Optional, Dict, Any, Mapping, Sequence
import math

import hashlib
import json

from packaging.version import Version

from ... import env as jit_env
from ...cpp_ext import get_mcc_version
from ...core import JitSpec, gen_jit_spec
from ....utils import ceil_div
from .fmha_utils import (
    _FP8_DTYPES,
    _get_fwd_kernel_config,
    fmha_extra_include_paths,
    get_fmha_template,
    _resolve_mask,
    round_multiple,
)
from .fmha_combine import _fmha_fwd_combine
from ...utils import (
    dtype_torch2mutlass_map,
    maybe_contiguous,
    TVM_HEADER,
    EXPORT_FUNC,
)
from ...configs import KernelConfigGraph, ParamSpec, domain_by_case
from ....execution_context import raise_complete_if_dry_run, is_fake_mode


kern_fwd = get_fmha_template("fwd_kern.j2")

FMHA_FWD_EXTRA_CUDA_CFLAGS = [
    "-Od3",
    "-O2",
    "-DNDEBUG",
    "-fno-strict-aliasing",
    "-fno-signed-zeros",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-mtgpu-load-cluster-mutation=1",
    "-mllvm",
    "--num-dwords-of-load-in-mutation=64",
]

_MCC_FMHA_WORKAROUND_CUTOFF = Version("5.2.0")


def _get_fmha_fwd_extra_cuda_cflags() -> list[str]:
    flags = list(FMHA_FWD_EXTRA_CUDA_CFLAGS)
    mcc_version = get_mcc_version()
    if mcc_version is not None and mcc_version < _MCC_FMHA_WORKAROUND_CUTOFF:
        flags.extend(("-fno-slp-vectorize", "-DMATE_FMHA_USE_SCALAR_EXP2"))
    return flags


def _fmha_fwd_encode(config: Mapping[str, object]) -> str:
    uri = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    return f"fmha_fwd_{uri}"


def _render_fmha_fwd_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = str(
        render_config.get("func_name") or _fmha_fwd_encode(render_config)
    )
    return (
        TVM_HEADER + kern_fwd.render(render_config) + EXPORT_FUNC.render(render_config)
    )


@functools.cache
def _load_fmha_fwd_module(frozen_config: tuple[tuple[str, object], ...]):
    return gen_fmha_fwd_spec(dict(frozen_config)).build_and_load()


def _fmha_fwd_module(config: Mapping[str, object]):
    dispatch_name = _fmha_fwd_encode(config)
    return dispatch_name, _load_fmha_fwd_module(tuple(sorted(config.items())))


def config_selector(cfg):
    return cfg["config_level"]


@functools.cache
def _dtype_bits(dtype: torch.dtype) -> int:
    return dtype.itemsize * 8


@functools.cache
def _paged_kv_tme_page_size(dtype: torch.dtype) -> int:
    return 128 if _dtype_bits(dtype) == 8 else 64


@functools.cache
def _select_lsu_load_kv(
    *,
    paged_kv: bool,
    page_size: int,
    has_leftpad_k: bool,
    has_qv: bool,
    dtype: torch.dtype,
) -> tuple[bool, bool]:
    tme_page_size = _paged_kv_tme_page_size(dtype)
    paged_tme_ok = paged_kv and page_size == tme_page_size and not has_leftpad_k
    use_lsu_load_k = not paged_tme_ok
    use_lsu_load_v = (paged_kv and not paged_tme_ok) or (has_qv and not paged_kv)
    return use_lsu_load_k, use_lsu_load_v


def _dtype_from_config(cfg: Mapping[str, object]) -> torch.dtype:
    dtype = cfg["dtype"]
    assert isinstance(dtype, torch.dtype)
    return dtype


def _select_lsu_load_kv_for_config(cfg: Mapping[str, object]) -> tuple[bool, bool]:
    dtype = _dtype_from_config(cfg)
    return _select_lsu_load_kv(
        paged_kv=bool(cfg["paged_kv"]),
        page_size=_paged_kv_tme_page_size(dtype),
        has_leftpad_k=bool(cfg["has_leftpad_k"]),
        has_qv=bool(cfg["has_qv"]),
        dtype=dtype,
    )


CONFIG_TABLE: Dict[int, Dict[str, Any]] = {
    1: {
        "dtype": [torch.bfloat16],
        "headdim": [(128, 128), (256, 256)],
        "head_ratio": [1, 4, 5, 8, 12, 16],
        "mode_q": ["ragged"],
        "mode_k": ["ragged", "padded"],
        "mode_mask": ["none", "causal"],
        "is_packgqa": [False, True],
        "has_metadata": [False, True],
        "has_cu_seqlens_k_new": [False],
        "has_leftpad_k": [False],
        "has_softcap": [False],
    },
    2: {
        "dtype": [torch.bfloat16],
        "headdim": [(128, 128), (192, 128), (256, 256)],
        "head_ratio": [1, 4, 5, 8, 12, 16],
        "mode_q": ["ragged"],
        "mode_k": ["normal", "ragged", "padded"],
        "mode_mask": ["none", "causal", "local"],
        "is_packgqa": [False, True],
        "has_metadata": [False, True],
        "has_cu_seqlens_k_new": [False],
        "has_leftpad_k": [False],
        "has_softcap": [False, True],
        "paged_kv": [False, True],
        "has_learnable_sink": [False, True],
        "enable_cp": [False, True],
    },
}

base_specs = []
mode_q = [
    ParamSpec(
        name="mode_q",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "mode_q"),
        default="normal",
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="has_cu_seqlens_q",
        default=False,
        compute=lambda cfg: cfg["mode_q"] == "ragged",
        depends_on=("mode_q",),
        sweep=False,
    ),
    ParamSpec(
        name="has_seqused_q",
        default=False,
        compute=lambda cfg: cfg["mode_q"] == "padded",
        depends_on=("mode_q",),
        sweep=False,
    ),
]
mode_k = [
    ParamSpec(
        name="mode_k",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "mode_k"),
        default="normal",
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="has_cu_seqlens_k",
        default=False,
        compute=lambda cfg: cfg["mode_k"] == "ragged",
        depends_on=("mode_k",),
        sweep=False,
    ),
    ParamSpec(
        name="has_seqused_k",
        default=False,
        compute=lambda cfg: cfg["mode_k"] == "padded",
        depends_on=("mode_k",),
        sweep=False,
    ),
    ParamSpec(
        name="paged_kv",
        domain=[True, False],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["has_seqused_k"]),
        depends_on=("has_seqused_k",),
    ),
    ParamSpec(
        name="use_lsu_load_k",
        compute=lambda cfg: _select_lsu_load_kv_for_config(cfg)[0],
        depends_on=(
            "dtype",
            "paged_kv",
            "has_leftpad_k",
            "has_qv",
        ),
        sweep=False,
    ),
    ParamSpec(
        name="use_lsu_load_v",
        compute=lambda cfg: _select_lsu_load_kv_for_config(cfg)[1],
        depends_on=(
            "dtype",
            "paged_kv",
            "has_leftpad_k",
            "has_qv",
        ),
        sweep=False,
    ),
    ParamSpec(
        name="is_append_kv",  # AppendKV not in AOT
        domain=[False],
        default=False,
    ),
    ParamSpec(
        name="has_cu_seqlens_k_new",
        domain=[False],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["is_append_kv"]),
        depends_on=("is_append_kv",),
    ),
    ParamSpec(
        name="has_leftpad_k",
        domain=[False, True],
        default=False,
    ),
    ParamSpec(  # NOTE:  Change me when supported.
        name="has_kv_batch_idx",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["has_seqused_k"]),
        depends_on=("has_seqused_k",),
    ),
]
mode_mask = [
    ParamSpec(
        name="mode_mask",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "mode_mask"),
        default="none",
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="is_causal",
        default=False,
        compute=lambda cfg: cfg["mode_mask"] == "causal",
        depends_on=("mode_mask",),
        sweep=False,
    ),
    ParamSpec(
        name="is_local",
        default=False,
        compute=lambda cfg: cfg["mode_mask"] == "local",
        depends_on=("mode_mask",),
        sweep=False,
    ),
    ParamSpec(
        name="has_attention_chunk",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["is_local"]),
        depends_on=("is_local",),
    ),
    ParamSpec(
        name="has_learnable_sink",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["is_local"]),
        depends_on=("is_local",),
    ),
]
mode_rope = [
    ParamSpec(  # No RoPE in AOT.
        name="is_rotary",
        domain=[
            False,
        ],
    ),
    ParamSpec(  # No RoPE in AOT.
        name="is_rotary_interleaved",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["is_rotary"]),
        depends_on=("is_rotary",),
    ),
    ParamSpec(  # No RoPE in AOT.
        name="has_seqlens_rotary",
        domain=[False, True],
        default=False,
        meaningful_if=lambda cfg: bool(cfg["is_rotary"]),
        depends_on=("is_rotary",),
    ),
]
score_mode = [
    ParamSpec(
        name="has_softcap",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "has_softcap"),
        depends_on=("config_level",),
    ),
]
specs_attn = [
    ParamSpec(
        name="has_qv",
        domain=[False],
        default=False,
    ),
    ParamSpec(
        name="only_qv",
        domain=[False],
        default=False,
    ),
    ParamSpec(
        name="is_packgqa",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "is_packgqa"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="dtype",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "dtype"),
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="element",
        compute=lambda cfg: dtype_torch2mutlass_map[_dtype_from_config(cfg)],
        depends_on=("dtype",),
        sweep=False,
    ),
    ParamSpec(
        name="has_q_descale",
        domain=[False],
        meaningful_if=lambda cfg: cfg["dtype"] in _FP8_DTYPES,
        depends_on=("dtype",),
        sweep=False,
    ),
    ParamSpec(
        name="has_k_descale",
        domain=[False],
        meaningful_if=lambda cfg: cfg["dtype"] in _FP8_DTYPES,
        depends_on=("dtype",),
        sweep=False,
    ),
    ParamSpec(
        name="has_v_descale",
        domain=[False],
        meaningful_if=lambda cfg: cfg["dtype"] in _FP8_DTYPES,
        depends_on=("dtype",),
        sweep=False,
    ),
    ParamSpec(
        name="head_ratio",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "head_ratio"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="has_metadata",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "has_metadata"),
        depends_on=("config_level",),
    ),
    ParamSpec(
        name="enable_cp",
        domain=[False],
        default=False,
    ),
]


def headdim_selector(cfg):
    hd = cfg["headdim"]
    if hd == (128, 128):
        return "128_128"
    elif hd == (192, 128):
        return "192_128"
    elif hd == (256, 256):
        return "256_256"
    # elif hd == (384, 384):
    #     return "384_384"
    else:
        raise ValueError(f"Unsupported headdim {hd}")


HEADDIM_TABLE: Dict[str, Dict[str, Any]] = {
    "128_128": {
        "tile_mn": [(32, 64), (64, 64), (128, 64), (192, 64), (256, 64)],
        "stages_kv": [(2, 2)],
    },
    "192_128": {
        "tile_mn": [(32, 64), (64, 64), (128, 64), (192, 64), (256, 64)],
        "stages_kv": [(1, 1)],
    },
    "256_256": {
        "tile_mn": [(32, 64), (192, 64)],
        "stages_kv": [(1, 1), (2, 2)],
    },
    "384_384": {
        "tile_mn": [(64, 64)],
        "stages_kv": [(1, 1)],
    },
}
specs_sel = [
    ParamSpec(
        name="headdim",
        domain=domain_by_case(config_selector, CONFIG_TABLE, "headdim"),
        default=(128, 128),
        depends_on=("config_level",),
        export=False,
    ),
    ParamSpec(
        name="head_dim_qk",
        compute=lambda cfg: cfg["headdim"][0],
        depends_on=("headdim",),
        sweep=False,
    ),
    ParamSpec(
        name="head_dim_vo",
        compute=lambda cfg: cfg["headdim"][1],
        depends_on=("headdim",),
        sweep=False,
    ),
    ParamSpec(
        name="is_even_headdim",
        compute=lambda cfg: cfg["headdim"][0] == cfg["headdim"][1],
        depends_on=("headdim",),
        sweep=False,
    ),
    ParamSpec(
        name="tile_mn",
        domain=domain_by_case(headdim_selector, HEADDIM_TABLE, "tile_mn"),
        default=(0, 0),
        depends_on=("headdim",),
        export=False,
    ),
    ParamSpec(
        name="tile_m",
        compute=lambda cfg: cfg["tile_mn"][0],
        depends_on=("tile_mn",),
        sweep=False,
    ),
    ParamSpec(
        name="tile_n",
        compute=lambda cfg: cfg["tile_mn"][1],
        depends_on=("tile_mn",),
        sweep=False,
    ),
    ParamSpec(
        name="stages_kv",
        domain=domain_by_case(headdim_selector, HEADDIM_TABLE, "stages_kv"),
        default=(0, 0),
        depends_on=("headdim",),
        export=False,
    ),
    ParamSpec(
        name="stages_k",
        compute=lambda cfg: cfg["stages_kv"][0],
        depends_on=("stages_kv",),
        sweep=False,
    ),
    ParamSpec(
        name="stages_v",
        compute=lambda cfg: cfg["stages_kv"][1],
        depends_on=("stages_kv",),
        sweep=False,
    ),
    ParamSpec(
        name="consumers_qk",
        compute=lambda cfg: ceil_div(cfg["tile_m"], 64),
        depends_on=("tile_m",),
        sweep=False,
    ),
    ParamSpec(
        name="consumers_pv",
        compute=lambda cfg: cfg["consumers_qk"],
        depends_on=("consumers_qk",),
        sweep=False,
    ),
]


base_specs.extend(mode_q)
base_specs.extend(mode_k)
base_specs.extend(mode_mask)
base_specs.extend(mode_rope)
base_specs.extend(score_mode)
base_specs.extend(specs_attn)
base_specs.extend(specs_sel)


def _gen_specs_from_config(cfg_level: int):
    config = [
        ParamSpec(
            name="config_level",
            domain=[cfg_level],
            default=1,
            export=False,
        ),
    ]

    specs = []
    specs.extend(base_specs)
    specs.extend(config)

    config_graph = KernelConfigGraph(specs)
    return config_graph.resolve_and_expand()


def get_fmha_fwd_aot_configs(config_level: int) -> list[dict[str, object]]:
    if config_level == 0:
        return []
    if config_level not in CONFIG_TABLE:
        raise ValueError(f"Unsupported FMHA fwd AOT level: {config_level}")
    return _gen_specs_from_config(config_level)


def gen_fmha_fwd_spec(config: Mapping[str, object]) -> JitSpec:
    dispatch_name = _fmha_fwd_encode(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu")
    return gen_jit_spec(
        name=dispatch_name,
        sources=[source_file],
        generated_sources={source_file: _render_fmha_fwd_source(config)},
        extra_cuda_cflags=_get_fmha_fwd_extra_cuda_cflags(),
        extra_include_paths=fmha_extra_include_paths(),
    )


def gen_fmha_fwd_specs(configs: Sequence[Mapping[str, object]]) -> list[JitSpec]:
    return [gen_fmha_fwd_spec(config) for config in configs]


def _fmha_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    q_v: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    learnable_sink: Optional[torch.Tensor] = None,
    softcap: float = 0.0,
    is_rotary_interleaved: bool = False,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    mp_margin: int = 0,
    return_lse: bool = False,
    lse: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    # CP params — default to non-CP (world_size=1)
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k: Optional[torch.Tensor] = None,
    only_qv: bool = False,
):
    # Feature gates.
    assert not ((k_new is None) ^ (v_new is None)), (
        "k_new and v_new must be provided together"
    )

    # Canonicalize tensor layout before deriving runtime metadata.
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    q_v = maybe_contiguous(q_v)
    if k_new is not None:
        k_new, v_new = [maybe_contiguous(t) for t in (k_new, v_new)]

    # Infer runtime shape metadata.
    num_head, head_dim = q.shape[-2:]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
        max_seqlen_q = q.shape[1]
    else:
        assert max_seqlen_q is not None
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = max_seqlen_q
        total_q = q.shape[0]

    if page_table is not None:
        max_num_pages_per_seq = page_table.shape[1]
        num_pages, page_size = k.shape[:2]
        seqlen_k = max_num_pages_per_seq * page_size
    else:
        num_pages, page_size = 0, 1
        seqlen_k = k.shape[-3]

    batch_size_k = (
        page_table.shape[0]
        if page_table is not None
        else (k.shape[0] if cu_seqlens_k is None else cu_seqlens_k.shape[0] - 1)
    )

    # Validate device placement and dtypes.
    assert all(
        t is None or t.is_musa
        for t in (
            q,
            k,
            v,
            q_v,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_k_new,
            seqused_q,
            seqused_k,
            page_table,
            learnable_sink,
            k_new,
            v_new,
            rotary_cos,
            rotary_sin,
            leftpad_k,
            kv_batch_idx,
            q_descale,
            k_descale,
            v_descale,
        )
    ), "inputs must be on MUSA device"

    assert q.dtype in (torch.float16, torch.bfloat16, *_FP8_DTYPES), (
        "inputs must be float16, bfloat16, float8_e4m3fn, or float8_e5m2"
    )
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    if q.dtype in (torch.float16, torch.bfloat16) and (
        head_dim % 2 != 0 or head_dim_v % 2 != 0
    ):
        raise ValueError(
            "FP16/BF16 Q/K and V head dimensions must be divisible by 2, "
            f"got Q/K={head_dim} and V={head_dim_v}"
        )
    if q.dtype in _FP8_DTYPES and (head_dim % 4 != 0 or head_dim_v % 4 != 0):
        raise ValueError(
            "FP8 Q/K and V head dimensions must be divisible by 4, "
            f"got Q/K={head_dim} and V={head_dim_v}"
        )
    if q_v is not None:
        assert q_v.dtype == q.dtype, "q_v must have the same dtype as q, k and v"
    if k_new is not None:
        assert k_new.dtype == q.dtype == v_new.dtype, (
            "k_new and v_new must have the same dtype as q, k and v"
        )

    for t in [
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        leftpad_k,
    ]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new, seqused_q, seqused_k, leftpad_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new, seqused_q, seqused_k, leftpad_k must be contiguous"
            )

    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous"

    if kv_batch_idx is not None:
        assert kv_batch_idx.is_musa, "kv_batch_idx must be on MUSA device"
        assert kv_batch_idx.dtype == torch.int32, "kv_batch_idx must be int32"
        assert kv_batch_idx.shape == (batch_size,), (
            "kv_batch_idx must have shape (batch_size,)"
        )
        assert kv_batch_idx.stride(0) == 1, "kv_batch_idx must be contiguous"

    if leftpad_k is not None:
        assert leftpad_k.is_musa, "leftpad_k must be on MUSA device"
        assert leftpad_k.dtype == torch.int32, "leftpad_k must be int32"
        assert leftpad_k.shape == (batch_size,), (
            "leftpad_k must have shape (batch_size,)"
        )
        assert leftpad_k.stride(0) == 1, "leftpad_k must be contiguous"

    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    # Validate old-KV and append-KV layouts.
    if kv_batch_idx is None:
        assert batch_size_k == batch_size, (
            "batch_size must equal batch_size_k when kv_batch_idx is None"
        )

    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size_k, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size_k, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
            assert page_table.shape == (batch_size_k, max_num_pages_per_seq)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )
        assert kv_batch_idx is None, (
            "kv_batch_idx is not supported when cu_seqlens_k is provided"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

    if q_v is not None:
        assert q_v.stride(-1) == 1, "q_v must have contiguous last dimension"
        expected_qv_shape = (
            (batch_size, seqlen_q, num_head, head_dim_v)
            if cu_seqlens_q is None
            else (total_q, num_head, head_dim_v)
        )
        assert q_v.shape == expected_qv_shape, (
            f"q_v must have shape {expected_qv_shape}, got {tuple(q_v.shape)}"
        )

    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )

    if k_new is not None:
        if cu_seqlens_k_new is None:
            seqlen_k_new = k_new.shape[1]
            assert k_new.shape == (batch_size, seqlen_k_new, num_head_kv, head_dim), (
                "k_new must have shape (batch_size, seqlen_k_new, num_head_kv, head_dim)"
            )
            assert v_new.shape == (batch_size, seqlen_k_new, num_head_kv, head_dim_v), (
                "v_new must have shape (batch_size, seqlen_k_new, num_head_kv, head_dim_v)"
            )
        else:
            total_k_new = k_new.shape[0]
            assert k_new.shape == (total_k_new, num_head_kv, head_dim), (
                "packed k_new must have shape (total_k_new, num_head_kv, head_dim)"
            )
            assert v_new.shape == (total_k_new, num_head_kv, head_dim_v), (
                "packed v_new must have shape (total_k_new, num_head_kv, head_dim_v)"
            )
            assert cu_seqlens_k_new.shape == (batch_size + 1,), (
                "cu_seqlens_k_new must have shape (batch_size + 1,)"
            )
            if not is_fake_mode():
                assert cu_seqlens_k_new[-1].item() == total_k_new, (
                    "cu_seqlens_k_new[-1] must equal total_k_new"
                )

    # RoPE
    if rotary_cos is not None:
        assert k_new is not None, (
            "If rotary cos/sin are provided, new key / value to be appended to KV cache must also be provided"
        )
        assert rotary_sin is not None, (
            "If rotary cos is provided, rotary sin must also be provided"
        )
        rotary_dim = rotary_cos.shape[1] * 2
        assert rotary_dim <= head_dim, "rotary_dim must be <= headdim"
        assert rotary_dim % 16 == 0, (
            "Only rotary dimensions divisible by 16 are currently supported"
        )

        seqlen_ro = rotary_cos.shape[0]
        if page_table is not None:
            assert seqlen_ro >= seqlen_k, (
                "cos/sin seqlen must be at least the seqlen of KV cache"
            )
        assert rotary_cos.dtype == rotary_sin.dtype == q.dtype, (
            "rotary_cos must have the same dtype as query"
        )
        assert rotary_cos.shape == rotary_sin.shape == (seqlen_ro, rotary_dim // 2)

        if seqlens_rotary is not None:
            assert seqlens_rotary.dtype == torch.int32, (
                "seqlens_rotary must have dtype torch.int32"
            )
            assert seqlens_rotary.shape == (batch_size,)

    if q_v is not None:
        assert head_dim_v <= 512, "q_v is only supported for value head dim <= 512"
        assert head_dim % 8 == 0, "q/k head dim should be a multiple of 8"
        assert head_dim_v % 8 == 0, "value head dim should be a multiple of 8"
        # assert q.dtype in (torch.float16, torch.bfloat16), (
        #     "q_v is only supported for fp16 and bf16 data type"
        # )
    if only_qv:
        assert q_v is not None, "only_qv=True requires q_v"
    # FP8
    if q.dtype in _FP8_DTYPES:
        if q_descale is not None:
            assert q_descale.shape == (batch_size, num_head_kv)
        if k_descale is not None:
            assert k_descale.shape == (batch_size, num_head_kv)
        if v_descale is not None:
            assert v_descale.shape == (batch_size, num_head_kv)

    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"

    # Derive execution policy.
    if learnable_sink is not None:
        # FIXME: bug learnable_sink + split
        num_splits = -1
        scheduler_metadata = None

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(
            head_dim_v if only_qv else head_dim + (head_dim_v if q_v is not None else 0)
        )

    qhead_per_kvhead = num_head // num_head_kv
    if attention_chunk > 0:
        fixed_seqlen = (
            cu_seqlens_q is None
            and cu_seqlens_k is None
            and cu_seqlens_k_new is None
            and seqused_q is None
            and seqused_k is None
            and leftpad_k is None
            and page_table is None
            and k_new is None
        )
        if fixed_seqlen and seqlen_q <= seqlen_k and attention_chunk >= seqlen_k:
            attention_chunk = 0

    is_causal, is_local, window_size_left, window_size_right = _resolve_mask(
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        attention_chunk=attention_chunk,
    )
    kernel_pack_gqa = pack_gqa

    # CP sanity checks
    assert cp_world_size > 0, (
        "cp_world_size must be positive, required by downstream unified code path. Use 1 if CP is not enabled."
    )
    assert cp_world_size != 1 or cp_rank == 0, (
        "When context parallelism is disabled, cp_rank must be zero"
    )
    assert cp_world_size == 1 or cp_tot_seqused_k is not None, (
        "cp_tot_seqused_k must be provided when context parallelism is enabled."
    )
    assert not (is_local and cp_world_size > 1), (
        "Local attention (sliding window) is not currently supported with context parallelism (cp_world_size > 1)."
    )

    out_torch_dtype = torch.bfloat16 if q.dtype in _FP8_DTYPES else q.dtype
    q_batch_seqlen_shape = (
        (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    )
    lse_shape = (
        (batch_size, num_head, seqlen_q)
        if cu_seqlens_q is None
        else (num_head, total_q)
    )

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=out_torch_dtype,
            device=q.device,
        )
    else:
        assert out.shape == (*q_batch_seqlen_shape, num_head, head_dim_v)
        assert out.dtype == out_torch_dtype
        assert out.device == q.device

    if lse is None:
        lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    else:
        assert lse.shape == lse_shape
        assert lse.dtype == torch.float32
        assert lse.device == q.device

    batch_rounded = round_multiple(batch_size, 4)
    # Metadata
    # If not provided, fallback to SingleTileScheduler.
    if scheduler_metadata is not None:
        assert scheduler_metadata.shape[0] >= batch_rounded * 4, (
            "metadata buffer is too small"
        )

    (
        tile_m,
        tile_n,
        stages_k,
        stages_v,
        headdim_rounded,
        headdim_v_rounded,
        consumers_qk,
        consumers_pv,
        enable_packgqa,
    ) = _get_fwd_kernel_config(
        max_seqlen_q,
        qhead_per_kvhead,
        head_dim,
        head_dim_v,
        q.element_size(),
        kernel_pack_gqa,
        q_v is not None,
        q.dtype in _FP8_DTYPES,
        is_local and attention_chunk != 0,
    )
    # print(
    #     f"{tile_m=}, {tile_n=}, {stages_k=}, {stages_v=}, {headdim_rounded=}, {headdim_v_rounded=}, {consumers_qk=}, {consumers_pv=}, {enable_packgqa=}"
    # )

    use_lsu_load_k, use_lsu_load_v = _select_lsu_load_kv(
        paged_kv=page_table is not None,
        page_size=page_size,
        has_leftpad_k=leftpad_k is not None,
        has_qv=q_v is not None,
        dtype=q.dtype,
    )
    constexpr_dict = {
        "has_cu_seqlens_q": cu_seqlens_q is not None,
        "has_cu_seqlens_k": cu_seqlens_k is not None,
        "has_seqused_q": seqused_q is not None,
        "has_seqused_k": seqused_k is not None,
        "has_cu_seqlens_k_new": cu_seqlens_k_new is not None,
        "has_kv_batch_idx": kv_batch_idx is not None,
        "has_leftpad_k": leftpad_k is not None,
        "has_q_descale": q_descale is not None,
        "has_k_descale": k_descale is not None,
        "has_v_descale": v_descale is not None,
        "paged_kv": page_table is not None,
        "has_softcap": softcap != 0.0,
        "is_append_kv": k_new is not None,
        "has_learnable_sink": learnable_sink is not None,
        "is_local": is_local,
        "is_causal": is_causal,
        "is_packgqa": enable_packgqa,
        "head_ratio": qhead_per_kvhead,
        "element": dtype_torch2mutlass_map[q.dtype],
        "use_lsu_load_k": use_lsu_load_k,
        "use_lsu_load_v": use_lsu_load_v,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "stages_k": stages_k,
        "stages_v": stages_v,
        "head_dim_qk": headdim_rounded,
        "head_dim_vo": headdim_v_rounded,
        "consumers_qk": consumers_qk,
        "consumers_pv": consumers_pv,
        "is_even_headdim": headdim_v_rounded == head_dim_v,
        "has_metadata": scheduler_metadata is not None,
        "enable_cp": cp_world_size > 1,
        "is_rotary": rotary_cos is not None,
        "is_rotary_interleaved": is_rotary_interleaved and rotary_cos is not None,
        "has_seqlens_rotary": seqlens_rotary is not None,
        "has_attention_chunk": attention_chunk > 0,
        "has_qv": q_v is not None,
        "only_qv": only_qv,
    }
    # print(f"tile_m: {tile_m}, tile_n: {tile_n}")

    dispatch_name, mod = _fmha_fwd_module(constexpr_dict)
    fmha_fwd_impl = mod.get_function(dispatch_name)

    # Short-circuit kernel call in fake mode.
    raise_complete_if_dry_run()

    accums, num_splits = fmha_fwd_impl(
        q,
        k,
        v,
        k_new,
        v_new,
        q_v,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        softcap,
        # is_rotary_interleaved,  # Not used
        mp_margin,
        num_splits,
        scheduler_metadata,
        learnable_sink,
        out,
        lse,
        # CP params — always passed, kernel ignores them when cp_world_size==1
        cp_world_size,
        cp_rank,
        cp_tot_seqused_k,
    )
    o_accum, lse_accum = accums

    _fmha_fwd_combine(
        out,
        lse,
        o_accum,
        lse_accum,
        tile_n,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        num_split=num_splits,
        metadata=scheduler_metadata,
    )

    return (out, lse, o_accum, lse_accum) if return_lse else (out, o_accum)


def gen_fmha_fwd_aot(config_level: int = 0) -> list[JitSpec]:
    return gen_fmha_fwd_specs(get_fmha_fwd_aot_configs(config_level))
