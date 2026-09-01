from __future__ import annotations

import functools
from typing import Literal, cast, overload

import torch
import torch.nn.functional as F


QuantRecipe = tuple[int, int, int, int]
_Operand = Literal["q", "k", "v"]
QuantizedPair = tuple[torch.Tensor, torch.Tensor]
QuantizedTriple = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
QuantizedResult = QuantizedPair | QuantizedTriple

_DEFAULT_QK_RECIPE: QuantRecipe = (128, 16, -1, 1)
_VALUE_BLOCK_RECIPE: QuantRecipe = (128, 128, -1, 128)
_SUPPORTED_RECIPES = {
    (-1, -1, -1, -1),
    (128, 128, -1, 1),
    _DEFAULT_QK_RECIPE,
    _VALUE_BLOCK_RECIPE,
}
_SUPPORTED_DTYPES = {torch.int8, torch.float8_e4m3fn}
_COMPILE_DISABLED: set[tuple[object, ...]] = set()


def _validate_operand(operand: str) -> _Operand:
    if operand not in {"q", "k", "v"}:
        raise ValueError(f"Unsupported SageAttention operand: {operand!r}.")
    return operand  # type: ignore[return-value]


def _validate_quant_recipe(quant_recipe: QuantRecipe) -> QuantRecipe:
    if quant_recipe not in _SUPPORTED_RECIPES:
        supported = ", ".join(str(item) for item in sorted(_SUPPORTED_RECIPES))
        raise ValueError(
            f"Unsupported SageAttention quant_recipe={quant_recipe}. Supported values: {supported}."
        )
    return quant_recipe


def _validate_quant_dtype(quant_dtype: torch.dtype) -> torch.dtype:
    if quant_dtype not in _SUPPORTED_DTYPES:
        supported = ", ".join(sorted(dtype.__repr__() for dtype in _SUPPORTED_DTYPES))
        raise ValueError(
            f"Unsupported SageAttention quant_dtype={quant_dtype}. Supported values: {supported}."
        )
    return quant_dtype


def _get_quant_bounds(quant_dtype: torch.dtype) -> tuple[float, float]:
    if quant_dtype == torch.int8:
        info = torch.iinfo(torch.int8)
        return float(info.max), float(info.min)
    info = torch.finfo(torch.float8_e4m3fn)
    return float(info.max), float(info.min)


def _quantize_full_tensor(
    x: torch.Tensor,
    *,
    quant_dtype: torch.dtype,
    return_dequant: bool,
) -> QuantizedResult:
    quant_max, quant_min = _get_quant_bounds(quant_dtype)
    amax = x.abs().to(torch.float32).amax().clamp(min=1e-12)
    scale = quant_max / amax
    q = (
        (x.to(torch.float32) * scale)
        .clamp(min=quant_min, max=quant_max)
        .to(quant_dtype)
    )
    scale_inv = scale.reciprocal().reshape(1, 1, 1, 1)
    if not return_dequant:
        return q, scale_inv
    return q, scale_inv, q.to(torch.float32) * scale_inv


def _quantize_sequence_groups(
    x_bhnd: torch.Tensor,
    *,
    group_size: int,
    quant_dtype: torch.dtype,
    return_dequant: bool,
) -> QuantizedResult:
    quant_max, quant_min = _get_quant_bounds(quant_dtype)
    batch, heads, seq_len, dim = x_bhnd.shape
    num_blocks = (seq_len + group_size - 1) // group_size
    padded_len = num_blocks * group_size
    if padded_len > seq_len:
        x_padded = F.pad(x_bhnd, [0, 0, 0, padded_len - seq_len, 0, 0, 0, 0])
    else:
        x_padded = x_bhnd
    x_blocks = x_padded.reshape(batch, heads, num_blocks, group_size, dim)
    amax = (
        x_blocks.abs().to(torch.float32).amax(dim=(3, 4), keepdim=True).clamp(min=1e-12)
    )
    scale = quant_max / amax
    q_blocks = (
        (x_blocks.to(torch.float32) * scale)
        .clamp(min=quant_min, max=quant_max)
        .to(quant_dtype)
    )
    q_padded = q_blocks.reshape(batch, heads, padded_len, dim)
    q = q_padded[:, :, :seq_len, :] if padded_len > seq_len else q_padded
    scale_inv = (
        scale.reciprocal()
        .reshape(batch, heads, num_blocks, 1)
        .transpose(1, 2)
        .contiguous()
    )
    if not return_dequant:
        return q, scale_inv
    dequant_padded = (q_blocks.to(torch.float32) * scale.reciprocal()).reshape(
        batch, heads, padded_len, dim
    )
    dequant = (
        dequant_padded[:, :, :seq_len, :] if padded_len > seq_len else dequant_padded
    )
    return q, scale_inv, dequant


def _quantize_sixteen_token_groups(
    x_bhnd: torch.Tensor,
    *,
    quant_dtype: torch.dtype,
    return_dequant: bool,
) -> QuantizedResult:
    quant_max, quant_min = _get_quant_bounds(quant_dtype)
    batch, heads, seq_len, dim = x_bhnd.shape
    block_n = 128
    num_blocks = (seq_len + block_n - 1) // block_n
    padded_len = num_blocks * block_n
    if padded_len > seq_len:
        x_padded = F.pad(x_bhnd, [0, 0, 0, padded_len - seq_len, 0, 0, 0, 0])
    else:
        x_padded = x_bhnd
    x_blocks = x_padded.reshape(batch, heads, num_blocks, 2, 8, 8, dim)
    amax = (
        x_blocks.abs()
        .to(torch.float32)
        .amax(dim=(3, 5, 6), keepdim=True)
        .clamp(min=1e-12)
    )
    scale = quant_max / amax
    q_blocks = (
        (x_blocks.to(torch.float32) * scale)
        .clamp(min=quant_min, max=quant_max)
        .to(quant_dtype)
    )
    q_padded = q_blocks.reshape(batch, heads, padded_len, dim)
    q = q_padded[:, :, :seq_len, :] if padded_len > seq_len else q_padded
    scale_inv = (
        scale.reciprocal()
        .reshape(batch, heads, padded_len // 16, 1)
        .transpose(1, 2)
        .contiguous()
    )
    if not return_dequant:
        return q, scale_inv
    dequant_padded = (q_blocks.to(torch.float32) * scale.reciprocal()).reshape(
        batch, heads, padded_len, dim
    )
    dequant = (
        dequant_padded[:, :, :seq_len, :] if padded_len > seq_len else dequant_padded
    )
    return q, scale_inv, dequant


def _quantize_sequence_channels(
    x_bnhd: torch.Tensor,
    *,
    quant_dtype: torch.dtype,
    return_dequant: bool,
) -> QuantizedResult:
    quant_max, quant_min = _get_quant_bounds(quant_dtype)
    amax = x_bnhd.abs().to(torch.float32).amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = quant_max / amax
    q = (
        (x_bnhd.to(torch.float32) * scale)
        .clamp(min=quant_min, max=quant_max)
        .to(quant_dtype)
    )
    scale_inv = scale.reciprocal()
    if not return_dequant:
        return q, scale_inv
    return q, scale_inv, q.to(torch.float32) * scale_inv


def _quantize_query_or_key(
    x_bnhd: torch.Tensor,
    *,
    operand: _Operand,
    quant_recipe: QuantRecipe,
    quant_dtype: torch.dtype,
    return_dequant: bool,
    smooth_k: bool,
) -> QuantizedResult:
    x_bhnd = x_bnhd.transpose(1, 2).contiguous()
    if operand == "k" and smooth_k:
        x_bhnd = x_bhnd - x_bhnd.mean(dim=2, keepdim=True)

    if not return_dequant:
        if quant_recipe == (-1, -1, -1, -1):
            q_bhnd, scale = cast(
                QuantizedPair,
                _quantize_full_tensor(
                    x_bhnd,
                    quant_dtype=quant_dtype,
                    return_dequant=False,
                ),
            )
        elif operand == "k" and quant_recipe == _DEFAULT_QK_RECIPE:
            q_bhnd, scale = cast(
                QuantizedPair,
                _quantize_sixteen_token_groups(
                    x_bhnd,
                    quant_dtype=quant_dtype,
                    return_dequant=False,
                ),
            )
        else:
            group_size = quant_recipe[0] if operand == "q" else quant_recipe[1]
            q_bhnd, scale = cast(
                QuantizedPair,
                _quantize_sequence_groups(
                    x_bhnd,
                    group_size=group_size,
                    quant_dtype=quant_dtype,
                    return_dequant=False,
                ),
            )
        return q_bhnd.transpose(1, 2).contiguous(), scale

    if quant_recipe == (-1, -1, -1, -1):
        q_bhnd, scale, dequant_bhnd = cast(
            QuantizedTriple,
            _quantize_full_tensor(
                x_bhnd,
                quant_dtype=quant_dtype,
                return_dequant=True,
            ),
        )
    elif operand == "k" and quant_recipe == _DEFAULT_QK_RECIPE:
        q_bhnd, scale, dequant_bhnd = cast(
            QuantizedTriple,
            _quantize_sixteen_token_groups(
                x_bhnd,
                quant_dtype=quant_dtype,
                return_dequant=True,
            ),
        )
    else:
        group_size = quant_recipe[0] if operand == "q" else quant_recipe[1]
        q_bhnd, scale, dequant_bhnd = cast(
            QuantizedTriple,
            _quantize_sequence_groups(
                x_bhnd,
                group_size=group_size,
                quant_dtype=quant_dtype,
                return_dequant=True,
            ),
        )
    return (
        q_bhnd.transpose(1, 2).contiguous(),
        scale,
        dequant_bhnd.transpose(1, 2).contiguous(),
    )


def _quantize_v(
    x_bnhd: torch.Tensor,
    *,
    quant_recipe: QuantRecipe,
    return_dequant: bool,
) -> QuantizedResult:
    if quant_recipe == _VALUE_BLOCK_RECIPE:
        if not return_dequant:
            q_bhnd, scale = cast(
                QuantizedPair,
                _quantize_sequence_groups(
                    x_bnhd.transpose(1, 2).contiguous(),
                    group_size=128,
                    quant_dtype=torch.float8_e4m3fn,
                    return_dequant=False,
                ),
            )
            return q_bhnd.transpose(1, 2).contiguous(), scale
        q_bhnd, scale, dequant_bhnd = cast(
            QuantizedTriple,
            _quantize_sequence_groups(
                x_bnhd.transpose(1, 2).contiguous(),
                group_size=128,
                quant_dtype=torch.float8_e4m3fn,
                return_dequant=True,
            ),
        )
        return (
            q_bhnd.transpose(1, 2).contiguous(),
            scale,
            dequant_bhnd.transpose(1, 2).contiguous(),
        )

    if quant_recipe == (-1, -1, -1, -1):
        if not return_dequant:
            q_bhnd, scale = cast(
                QuantizedPair,
                _quantize_full_tensor(
                    x_bnhd.transpose(1, 2).contiguous(),
                    quant_dtype=torch.float8_e4m3fn,
                    return_dequant=False,
                ),
            )
            return q_bhnd.transpose(1, 2).contiguous(), scale
        q_bhnd, scale, dequant_bhnd = cast(
            QuantizedTriple,
            _quantize_full_tensor(
                x_bnhd.transpose(1, 2).contiguous(),
                quant_dtype=torch.float8_e4m3fn,
                return_dequant=True,
            ),
        )
        return (
            q_bhnd.transpose(1, 2).contiguous(),
            scale,
            dequant_bhnd.transpose(1, 2).contiguous(),
        )

    return _quantize_sequence_channels(
        x_bnhd,
        quant_dtype=torch.float8_e4m3fn,
        return_dequant=return_dequant,
    )


def _quantize_sage_attention_tensor_eager(
    x: torch.Tensor,
    *,
    operand: _Operand,
    quant_recipe: QuantRecipe,
    quant_dtype: torch.dtype,
    return_dequant: bool,
    smooth_k: bool,
) -> QuantizedResult:
    if operand == "v":
        if quant_dtype != torch.float8_e4m3fn:
            raise ValueError(
                "SageAttention value quantization only supports torch.float8_e4m3fn."
            )
        return _quantize_v(x, quant_recipe=quant_recipe, return_dequant=return_dequant)

    return _quantize_query_or_key(
        x,
        operand=operand,
        quant_recipe=quant_recipe,
        quant_dtype=quant_dtype,
        return_dequant=return_dequant,
        smooth_k=smooth_k,
    )


@functools.lru_cache(maxsize=None)
def _get_compiled_quantizer(
    operand: _Operand,
    quant_recipe: QuantRecipe,
    quant_dtype: torch.dtype,
    return_dequant: bool,
    smooth_k: bool,
):
    if not hasattr(torch, "compile"):
        return None

    def compiled_quantizer(x: torch.Tensor):
        return _quantize_sage_attention_tensor_eager(
            x,
            operand=operand,
            quant_recipe=quant_recipe,
            quant_dtype=quant_dtype,
            return_dequant=return_dequant,
            smooth_k=smooth_k,
        )

    try:
        return torch.compile(compiled_quantizer, dynamic=True, fullgraph=False)
    except Exception:
        return None


@overload
def quantize_sage_attention_tensor(
    x: torch.Tensor,
    *,
    operand: str,
    quant_recipe: QuantRecipe,
    quant_dtype: torch.dtype,
    return_dequant: Literal[False] = False,
    smooth_k: bool = False,
    use_compile: bool = True,
) -> QuantizedPair: ...


@overload
def quantize_sage_attention_tensor(
    x: torch.Tensor,
    *,
    operand: str,
    quant_recipe: QuantRecipe,
    quant_dtype: torch.dtype,
    return_dequant: Literal[True],
    smooth_k: bool = False,
    use_compile: bool = True,
) -> QuantizedTriple: ...


def quantize_sage_attention_tensor(
    x: torch.Tensor,
    *,
    operand: str,
    quant_recipe: QuantRecipe,
    quant_dtype: torch.dtype,
    return_dequant: bool = False,
    smooth_k: bool = False,
    use_compile: bool = True,
) -> QuantizedResult:
    operand = _validate_operand(operand)
    quant_recipe = _validate_quant_recipe(quant_recipe)
    quant_dtype = _validate_quant_dtype(quant_dtype)
    compile_key = (operand, quant_recipe, quant_dtype, return_dequant, smooth_k)

    if use_compile and compile_key not in _COMPILE_DISABLED:
        compiled_quantizer = _get_compiled_quantizer(
            operand,
            quant_recipe,
            quant_dtype,
            return_dequant,
            smooth_k,
        )
        if compiled_quantizer is not None:
            try:
                return compiled_quantizer(x)
            except Exception:
                _COMPILE_DISABLED.add(compile_key)

    return _quantize_sage_attention_tensor_eager(
        x,
        operand=operand,
        quant_recipe=quant_recipe,
        quant_dtype=quant_dtype,
        return_dequant=return_dequant,
        smooth_k=smooth_k,
    )


__all__ = ["quantize_sage_attention_tensor"]
