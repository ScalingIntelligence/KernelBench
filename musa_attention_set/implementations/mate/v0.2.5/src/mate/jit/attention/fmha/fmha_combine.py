from functools import lru_cache
import functools
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from ... import env as jit_env
from ...core import JitSpec, gen_jit_spec
from ...utils import dtype_torch2mutlass_map, TVM_HEADER, EXPORT_FUNC
from ...configs import KernelConfigGraph, ParamSpec
from .fmha_utils import (
    FMHA_EXTRA_CUDA_CFLAGS,
    fmha_extra_include_paths,
    get_fmha_template,
)
from ....execution_context import raise_complete_if_dry_run


def _fmha_fwd_combine_encode(config: Mapping[str, object]) -> str:
    name_list = ["fmha_fwd_combine"]
    if config["element"] == "mutlass::half_t":
        name_list.append("f16")
    elif config["element"] == "mutlass::bfloat16_t":
        name_list.append("bf16")
    elif config["element"] == "float":
        name_list.append("f32")
    else:
        raise ValueError(f"Unsupported element type: {config['element']}")

    name_list.append(f"{config['tile_m']}x{config['tile_n']}x{config['max_splits']}")
    if config["has_cu_seqlens_q"]:
        mode = "ragged_q"
        name_list.append(mode)
    elif config["has_seqused_q"]:
        mode = "padded_q"
        name_list.append(mode)
    if config["has_metadata"]:
        name_list.append("metadata")
    return "_".join(name_list)


def _render_fmha_fwd_combine_source(config: Mapping[str, object]) -> str:
    render_config = dict(config)
    render_config["func_name"] = str(
        render_config.get("func_name") or _fmha_fwd_combine_encode(render_config)
    )
    return (
        TVM_HEADER
        + get_fmha_template("combine_kern.j2").render(render_config)
        + EXPORT_FUNC.render(render_config)
    )


@lru_cache
def _get_fwd_combine_kernel_config(tile_n: int, num_split: int):
    assert tile_n % 32 == 0, "tile_n must be multiple of 32"
    if tile_n % 128 == 0:
        tile_m = 8
    elif tile_n % 64 == 0:
        tile_m = 16
    else:
        tile_m = 32

    if tile_m >= 16 and num_split <= 16:
        max_splits = 16
    elif num_split <= 32:
        max_splits = 32
    elif num_split <= 64:
        max_splits = 64
    elif num_split <= 128:
        max_splits = 128
    elif num_split <= 256:
        max_splits = 256
    else:
        raise ValueError("num_split exceeds max supported splits 256")

    return tile_m, max_splits


def _fmha_fwd_combine_module(config: Mapping[str, object]):
    dispatch_name = _fmha_fwd_combine_encode(config)
    return dispatch_name, _load_fmha_fwd_combine_module(tuple(sorted(config.items())))


@functools.cache
def _load_fmha_fwd_combine_module(frozen_config: tuple[tuple[str, object], ...]):
    return gen_fmha_fwd_combine_spec(dict(frozen_config)).build_and_load()


specs = []
mode_q = [
    ParamSpec(
        name="mode_q",
        domain=["padded", "ragged", "normal"],
        default="normal",
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
specs.append(
    ParamSpec(
        name="element",
        domain=[dtype_torch2mutlass_map[x] for x in [torch.bfloat16]],
    )
)
specs_select = [
    ParamSpec(
        name="has_metadata",
        domain=[False, True],
    ),
    ParamSpec(
        name="tile_n",
        domain=[64],
    ),
    ParamSpec(
        name="tile_m",
        domain=[16],
    ),
    ParamSpec(
        name="max_splits",
        domain=[16, 32, 64, 128, 256],
    ),
]
specs.extend(mode_q)
specs.extend(specs_select)
aot_combine_configs = KernelConfigGraph(specs).resolve_and_expand()


def get_fmha_fwd_combine_aot_configs(config_level: int) -> list[dict[str, object]]:
    if config_level != 1:
        return []
    return aot_combine_configs


def gen_fmha_fwd_combine_spec(config: Mapping[str, object]) -> JitSpec:
    dispatch_name = _fmha_fwd_combine_encode(config)
    source_file = Path(jit_env.MATE_GEN_SRC_DIR / f"{dispatch_name}.mu")
    return gen_jit_spec(
        name=dispatch_name,
        sources=[source_file],
        generated_sources={source_file: _render_fmha_fwd_combine_source(config)},
        extra_cuda_cflags=list(FMHA_EXTRA_CUDA_CFLAGS),
        extra_include_paths=fmha_extra_include_paths(),
    )


def gen_fmha_fwd_combine_specs(
    configs: Sequence[Mapping[str, object]],
) -> list[JitSpec]:
    return [gen_fmha_fwd_combine_spec(config) for config in configs]


def gen_fmha_fwd_combine_aot(config_level: int = 0) -> list[JitSpec]:
    return gen_fmha_fwd_combine_specs(get_fmha_fwd_combine_aot_configs(config_level))


def _fmha_fwd_combine(
    out: torch.Tensor,
    lse: torch.Tensor,
    out_accum: torch.Tensor,
    lse_accum: torch.Tensor,
    tile_n: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    num_split: int = 0,
    metadata: Optional[torch.Tensor] = None,
) -> None:
    # Exit in dry run because combine is AOT.
    raise_complete_if_dry_run()

    if metadata is None and num_split <= 1:
        return

    if metadata is not None:
        if cu_seqlens_q is None:
            batch_size = out_accum.shape[0]
        else:
            assert max_seqlen_q is not None
            batch_size = cu_seqlens_q.shape[0] - 1

        assert metadata.shape[0] >= batch_size * 4, "metadata buffer is too small"

    tile_m, max_splits = _get_fwd_combine_kernel_config(tile_n, num_split=num_split)
    constexpr_dict = {
        "has_cu_seqlens_q": cu_seqlens_q is not None,
        "has_seqused_q": seqused_q is not None,
        "has_metadata": metadata is not None,
        "element": dtype_torch2mutlass_map[out.dtype],
        "tile_m": tile_m,
        "tile_n": tile_n,
        "max_splits": max_splits,
    }

    dispatch_name, mod = _fmha_fwd_combine_module(constexpr_dict)
    fmha_fwd_combine_impl = mod.get_function(dispatch_name)
    fmha_fwd_combine_impl(
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        out,
        lse,
        out_accum,
        lse_accum,
        metadata,
        num_split,
    )


def _flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise_complete_if_dry_run()

    if out_partial.dim() != 5:
        raise ValueError("out_partial must be a 5D tensor")
    if lse_partial.dim() != 4:
        raise ValueError("lse_partial must be a 4D tensor")
    if out_partial.dtype != torch.float32:
        raise ValueError("out_partial must be float32")
    if lse_partial.dtype != torch.float32:
        raise ValueError("lse_partial must be float32")
    if out_partial.device != lse_partial.device:
        raise ValueError("out_partial and lse_partial must be on the same device")
    if out_partial.stride(-1) != 1:
        raise ValueError("out_partial must have contiguous last dimension")
    if lse_partial.stride(-2) != 1:
        raise ValueError("lse_partial must be contiguous in the seqlen dimension")

    num_splits, batch_size, seqlen_q, num_head, headdim_v = out_partial.shape
    if num_splits <= 0:
        raise ValueError("out_partial must have at least one split")
    if lse_partial.shape != (num_splits, batch_size, seqlen_q, num_head):
        raise ValueError("lse_partial has unexpected shape")

    out_dtype = out_dtype or out_partial.dtype
    if out_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("out_dtype must be float32, float16, or bfloat16")

    if out is None:
        out = torch.empty(
            (batch_size, seqlen_q, num_head, headdim_v),
            device=out_partial.device,
            dtype=out_dtype,
        )
    else:
        if out.shape != (batch_size, seqlen_q, num_head, headdim_v):
            raise ValueError("out has unexpected shape")
        if out.dtype != out_dtype:
            raise ValueError("out dtype must match out_dtype")
        if out.device != out_partial.device:
            raise ValueError("out must be on the same device as out_partial")
        if out.stride(-1) != 1:
            raise ValueError("out must have contiguous last dimension")

    lse_storage = torch.empty(
        (batch_size, num_head, seqlen_q),
        device=out_partial.device,
        dtype=torch.float32,
    )
    lse = lse_storage.transpose(1, 2)

    if num_splits == 1:
        out.copy_(out_partial[0].to(dtype=out.dtype))
        lse.copy_(lse_partial[0])
        out.masked_fill_(~torch.isfinite(lse).unsqueeze(-1), 0)
        return out, lse
    if batch_size == 0 or seqlen_q == 0:
        return out, lse

    out_partial_for_kernel = out_partial.transpose(2, 3)
    lse_partial_for_kernel = lse_partial.transpose(2, 3)
    out_for_kernel = out
    padded_headdim_v = ((headdim_v + 3) // 4) * 4
    if padded_headdim_v != headdim_v:
        padded_out_partial = torch.empty(
            (num_splits, batch_size, num_head, seqlen_q, padded_headdim_v),
            device=out_partial.device,
            dtype=out_partial.dtype,
        )
        padded_out_partial[..., :headdim_v].copy_(out_partial_for_kernel)
        padded_out_partial[..., headdim_v:].zero_()
        out_partial_for_kernel = padded_out_partial
        out_for_kernel = torch.empty(
            (batch_size, seqlen_q, num_head, padded_headdim_v),
            device=out_partial.device,
            dtype=out.dtype,
        )

    _fmha_fwd_combine(
        out_for_kernel,
        lse,
        out_partial_for_kernel,
        lse_partial_for_kernel,
        64,
        num_split=num_splits,
        metadata=None,
    )
    if out_for_kernel is not out:
        out.copy_(out_for_kernel[..., :headdim_v])
    return out, lse
