import functools
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import EXPORT_FUNC, TVM_HEADER, dtype_torch2mutlass_map
from ...configs import KernelConfigGraph, ParamSpec, domain_by_case
from .fmha_utils import (
    _ELEMENT_NAME_SUFFIX,
    _FP8_DTYPES,
    FMHA_EXTRA_CUDA_CFLAGS,
    _get_fwd_kernel_config as _get_metadata_kernel_config,
    fmha_extra_include_paths,
    get_fmha_template,
    _resolve_mask,
    round_multiple,
)
from ....execution_context import raise_complete_if_dry_run


def _fmha_get_metadata_encode(config: Mapping[str, object]) -> str:
    name_list = ["fmha_get_metadata"]
    head_ratio = config["head_ratio"]
    num_warps = config["num_warps"]
    name_list.append(f"{head_ratio}x{num_warps}")
    name_list.append(_ELEMENT_NAME_SUFFIX[str(config["element"])])

    if config["has_seqused_q"]:
        mode_q = "padded_q"
        name_list.append(mode_q)
    elif config["has_cu_seqlens_q"]:
        mode_q = "ragged_q"
        name_list.append(mode_q)

    if config["has_seqused_k"]:
        mode_k = "padded_k"
        name_list.append(mode_k)
    elif config["has_cu_seqlens_k"]:
        mode_k = "ragged_k"
        name_list.append(mode_k)

    if config["has_cu_seqlens_k_new"]:
        mode_k_new = "ragged_knew"
        name_list.append(mode_k_new)

    if config["has_leftpad_k"]:
        name_list.append("leftpad_k")

    if config["is_causal"]:
        mode_causal = "causal"
        name_list.append(mode_causal)
    elif config["is_local"]:
        mode_local = "local"
        name_list.append(mode_local)
    if config["is_packgqa"]:
        mode_packgqa = "packgqa"
        name_list.append(mode_packgqa)

    return "_".join(name_list)


def _render_fmha_metadata_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = str(
        render_config.get("func_name") or _fmha_get_metadata_encode(render_config)
    )
    return (
        TVM_HEADER
        + get_fmha_template("metadata_kern.j2").render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


def _fmha_metadata_module(config: Mapping[str, object]):
    dispatch_name = _fmha_get_metadata_encode(config)
    return dispatch_name, _load_fmha_metadata_module(tuple(sorted(config.items())))


@functools.cache
def _load_fmha_metadata_module(frozen_config: tuple[tuple[str, object], ...]):
    return gen_fmha_metadata_spec(dict(frozen_config)).build_and_load()


def config_selector(cfg):
    return cfg["config_level"]


_CONFIG_TABLE: Dict[int, Dict[str, Any]] = {
    1: {
        "mode_q": ["ragged"],
        "mode_k": ["ragged", "padded"],
    },
    2: {
        "mode_q": ["normal", "ragged"],
        "mode_k": ["normal", "ragged", "padded"],
    },
}

base_specs = []
mode_q = [
    ParamSpec(
        name="mode_q",
        domain=domain_by_case(config_selector, _CONFIG_TABLE, "mode_q"),
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
        domain=domain_by_case(config_selector, _CONFIG_TABLE, "mode_k"),
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
        name="has_cu_seqlens_k_new",
        domain=[False],
        default=False,
    ),
    ParamSpec(
        name="has_leftpad_k",
        domain=[False, True],
        default=False,
    ),
]
mode_mask = [
    ParamSpec(
        name="mode_mask",
        domain=["none", "causal", "local"],
        default="none",
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
]
specs_attn = [
    ParamSpec(
        name="is_packgqa",
        domain=[False, True],
    ),
    ParamSpec(
        name="head_ratio",
        domain=[1, 2, 4, 5, 8, 12, 16],
    ),
    ParamSpec(
        name="element",
        domain=["mutlass::half_t", "mutlass::bfloat16_t"],
    ),
]
specs_metadata = [
    ParamSpec(
        name="element_size",
        compute=lambda cfg: torch.bfloat16.itemsize,
        sweep=False,
    ),
    ParamSpec(
        name="sort",
        domain=[True],
        default=True,
    ),
    ParamSpec(
        name="num_warps",
        domain=list(range(1, 6)),
        default=1,
    ),
]
base_specs.extend(mode_q)
base_specs.extend(mode_k)
base_specs.extend(mode_mask)
base_specs.extend(specs_attn)
base_specs.extend(specs_metadata)


def _gen_specs_from_config(cfg_level: int):
    config = [
        ParamSpec(
            name="config_level",
            domain=[cfg_level],
            default=1,
            export=False,
        ),
    ]
    config_graph = KernelConfigGraph(base_specs + config)
    return config_graph.resolve_and_expand()


def get_fmha_metadata_aot_configs(config_level: int) -> list[dict[str, object]]:
    if config_level == 0:
        return []
    if config_level not in _CONFIG_TABLE:
        raise ValueError(f"Unsupported FMHA metadata AOT level: {config_level}")
    return _gen_specs_from_config(config_level)


def gen_fmha_metadata_spec(config: Mapping[str, object]) -> JitSpec:
    dispatch_name = _fmha_get_metadata_encode(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu")
    return gen_jit_spec(
        name=dispatch_name,
        sources=[source_file],
        generated_sources={source_file: _render_fmha_metadata_source(config)},
        extra_cuda_cflags=list(FMHA_EXTRA_CUDA_CFLAGS),
        extra_include_paths=fmha_extra_include_paths(),
    )


def gen_fmha_metadata_specs(
    configs: Sequence[Mapping[str, object]],
) -> list[JitSpec]:
    return [gen_fmha_metadata_spec(config) for config in configs]


def gen_fmha_metadata_aot(config_level: int = 0) -> list[JitSpec]:
    return gen_fmha_metadata_specs(get_fmha_metadata_aot_configs(config_level))


def _fmha_get_metadata(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    max_seqlen_k_new: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: Optional[int],
    qkv_dtype: torch.dtype = torch.bfloat16,
    has_qv: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    leftpad_k: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    packgqa: Optional[bool] = None,
    mp_margin: int = 0,
) -> torch.Tensor:
    if cu_seqlens_q is not None:
        assert max_seqlen_q is not None

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )
    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert leftpad_k is None or leftpad_k.shape == (batch_size,), (
        "leftpad_k must have shape (batch_size,)"
    )

    for tensor in [
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        leftpad_k,
    ]:
        if tensor is not None:
            assert tensor.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, leftpad_k must be int32"
            )
            assert tensor.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, leftpad_k must be contiguous"
            )

    assert all(
        tensor is None or tensor.is_musa
        for tensor in (
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            leftpad_k,
        )
    ), "inputs must be on MUSA device"

    assert num_heads_q % num_heads_kv == 0, (
        "num_heads_q must be divisible by num_heads_kv"
    )
    if attention_chunk > 0:
        fixed_seqlen = (
            cu_seqlens_q is None
            and cu_seqlens_k is None
            and cu_seqlens_k_new is None
            and seqused_q is None
            and seqused_k is None
        )
        if (
            fixed_seqlen
            and max_seqlen_q <= max_seqlen_k
            and attention_chunk >= max_seqlen_k
        ):
            attention_chunk = 0
    is_causal, is_local, window_size_left, window_size_right = _resolve_mask(
        seqlen_q=max_seqlen_q,
        seqlen_k=max_seqlen_k,
        is_causal=is_causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        attention_chunk=attention_chunk,
    )
    qhead_per_kvhead = num_heads_q // num_heads_kv
    kernel_packgqa = packgqa
    element_size = qkv_dtype.itemsize

    batch_rounded = round_multiple(batch_size, 4)
    metadata = torch.empty((batch_rounded * 4), dtype=torch.int32, device="musa")

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
    ) = _get_metadata_kernel_config(
        max_seqlen_q,
        qhead_per_kvhead,
        headdim,
        headdim_v,
        element_size,
        kernel_packgqa,
        has_qv,
        qkv_dtype in _FP8_DTYPES,
    )

    packgqa = enable_packgqa if kernel_packgqa is None else kernel_packgqa
    # num_warps = 1 << (math.ceil(batch_size / 31) - 1).bit_length()
    num_warps = min(math.ceil(batch_size / 31), 32)
    constexpr_dict = {
        "has_cu_seqlens_q": cu_seqlens_q is not None,
        "has_cu_seqlens_k": cu_seqlens_k is not None,
        "has_seqused_q": seqused_q is not None,
        "has_seqused_k": seqused_k is not None,
        "has_cu_seqlens_k_new": cu_seqlens_k_new is not None,
        "has_leftpad_k": leftpad_k is not None,
        "is_causal": is_causal,
        "is_local": is_local,
        "is_packgqa": packgqa,
        "head_ratio": qhead_per_kvhead,
        "element_size": element_size,
        "sort": True,
        "num_warps": num_warps,
        "element": dtype_torch2mutlass_map[qkv_dtype],
    }

    dispatch_name, mod = _fmha_metadata_module(constexpr_dict)
    fmha_metadata_impl = mod.get_function(dispatch_name)

    # Exit in dry run
    raise_complete_if_dry_run()

    fmha_metadata_impl(
        batch_size,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        max_seqlen_q,
        max_seqlen_k,
        max_seqlen_k_new,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cu_seqlens_k_new,
        window_size_left,
        window_size_right,
        leftpad_k,
        metadata,
        num_splits,
        tile_m,
        tile_n,
        mp_margin,
    )
    return metadata
