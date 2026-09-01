import functools
from enum import Enum
from typing import Literal, Optional, Tuple, Union, cast

import torch

from mate.api_logging import mate_api
from mate._backend import resolve_backend
from mate.jit.gemm.deep_gemm.gemm import (
    GEMM_TYPE_M_GROUPED_CONTIGUOUS,
    GEMM_TYPE_M_GROUPED_MASKED,
    GEMM_TYPE_NORMAL,
    get_deep_gemm_gemm_module,
)
from mate.jit.gemm.masked_moe_gemm_mixed_dtype import (
    masked_moe_gemm_mixed_dtype_mutlass,
)
from mate.jit.gemm_ops import get_gemm_ops_module
from mate.jit.mubin.gemm import (
    groupwise_gemm_8bit_fp8output_mubin,
    m_grouped_contig_gemm_16bit_mubin,
    m_grouped_contig_gemm_8bit_mubin,
    masked_moe_gemm_16bit_mubin,
    masked_moe_gemm_8bit_mubin,
    masked_moe_gemm_w4a8_mubin,
    ragged_k_moe_gemm_16bit_mubin,
    ragged_k_moe_gemm_8bit_mubin,
    ragged_moe_gemm_16bit_mubin,
    ragged_moe_gemm_8bit_mubin,
    ragged_moe_gemm_w4a8_mubin,
)
from mate.mate_runtime import resolve_num_mps
from mate.utils import ceil_div


class GemmMixedDType(str, Enum):
    S4FP8 = "s4fp8"
    FP4FP8 = "fp4fp8"


def _resolve_gemm_mixed_dtype(
    mixed_dtype: GemmMixedDType | str,
) -> GemmMixedDType:
    if not isinstance(mixed_dtype, GemmMixedDType):
        try:
            mixed_dtype = GemmMixedDType(mixed_dtype)
        except ValueError as exc:
            allowed = [item.value for item in GemmMixedDType]
            raise ValueError(f"mixed_dtype must be one of {allowed}") from exc

    return mixed_dtype


def _resolve_moe_gemm_quant_recipe(
    name: str,
    quant_recipe: object,
) -> Tuple[int, int]:
    if (
        not isinstance(quant_recipe, tuple)
        or len(quant_recipe) != 2
        or any(type(item) is not int for item in quant_recipe)
    ):
        raise TypeError(f"{name} must be a tuple of two int values")

    return cast(Tuple[int, int], quant_recipe)


_W4A8_MUTLASS_A_QUANT_RECIPES = ((1, -1), (1, 128))


def check_w4a8_mutlass(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    a_quant_recipe: Tuple[int, int],
    enable_overlap: bool,
) -> bool:
    if enable_overlap or a_quant_recipe not in _W4A8_MUTLASS_A_QUANT_RECIPES:
        return False

    a, scale_a = input_a
    b, scale_b = input_b

    tensors = (a, scale_a, b, scale_b, masked_tokens_info, out)
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        return False
    if not (
        a.ndim == scale_a.ndim == b.ndim == scale_b.ndim == out.ndim == 3
        and masked_tokens_info.ndim == 1
    ):
        return False

    groups, max_m, k = map(int, a.shape)
    n = int(b.shape[1])
    scale_a_k_blocks = 1 if a_quant_recipe == (1, -1) else ceil_div(k, 128)
    contiguous_tensors = (a, b, scale_b, masked_tokens_info, out)
    scale_a_layout_supported = a_quant_recipe == (1, -1) or (
        scale_a.stride(-1) == 1 or scale_a.stride(1) == 1
    )
    return (
        all(tensor.device == a.device for tensor in tensors)
        and all(tensor.stride(-1) == 1 for tensor in contiguous_tensors)
        and scale_a_layout_supported
        and a.dtype == torch.float8_e4m3fn
        and scale_a.dtype == torch.float32
        and b.dtype == torch.int8
        and scale_b.dtype == torch.bfloat16
        and masked_tokens_info.dtype == torch.int32
        and out.dtype in (torch.float16, torch.bfloat16)
        and tuple(scale_a.shape) == (groups, max_m, scale_a_k_blocks)
        and tuple(b.shape) == (groups, n, ceil_div(k, 2))
        and tuple(scale_b.shape) == (groups, n, ceil_div(k, 128))
        and tuple(masked_tokens_info.shape) == (groups,)
        and tuple(out.shape) == (groups, max_m, n)
    )


@mate_api
def ragged_m_moe_gemm_16bit(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[
        Literal["per_token", "psum_expert", "per_expert"]
    ] = "per_token",
    major_a_mode: Optional[Literal["M", "K"]] = "K",
    major_b_mode: Optional[Literal["N", "K"]] = "K",
    num_mp: Optional[int] = None,
    alignment_m: Optional[int] = None,
    backend: Optional[Literal["auto", "mubin", "mutlass"]] = "auto",
):
    """
    Perform 16-bit GEMM operation for MoE (Mixture of Experts) with ragged tensor inputs.

    This function computes matrix multiplication between 16-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens assigned to them.

    Parameters
    ----------
    input_a : Tensor
        Input tensor A with shape ``(total_tokens, hidden_size)`` in fp16/bf16 format.
    input_b : Tensor
        Input tensor B with shape ``(num_expert, out_hidden_size, hidden_size)`` in fp16/bf16 format.
    ragged_tokens_info : Tensor
        If gemm_mode is `per_token`:
            Tensor indicating which expert each token belongs to, with shape ``(total_tokens,)``.
            Values represent expert indices, with -1 for unused positions.
        If gemm_mode is `psum_expert`
            Tensor with shape `(num_expert, )`, indicating how many tokens that first few experts have.
        If gemm_mode is `per_expert`
            Tensor with shape `(num_expert, )`, indicating how many tokens that every expert has.
    out : Tensor
        Output tensor with shape ``(total_tokens, out_hidden_size)``.
    major_a_mode : Optional[str]
        Indicating major stride of A.
        Default to `K`.
    major_b_mode : Optional[str]
        Indicating major stride of B.
        Default to `K`.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    alignment_m : Optional[int]
        Alignment requirement for total_tokens (m) dimension. Must be 128 or 256.
        Default is 128.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Tensor
        Result tensor with shape ``(total_tokens, out_hidden_size)`` containing the GEMM output in fp16 or bf16 data type.

    """

    if alignment_m is None:
        alignment_m = 128

    backend = cast(
        Literal["auto", "mubin", "mutlass"],
        resolve_backend(backend, supported=("mubin", "mutlass"), default="auto"),
    )

    if gemm_mode == "per_token":
        if backend == "mutlass":
            dispatch_name, mod = get_deep_gemm_gemm_module(
                kind="bf16",
                gemm_type=GEMM_TYPE_M_GROUPED_CONTIGUOUS,
                config_m=input_a.shape[0],
                alignment_m=alignment_m,
            )
            mod.get_function(dispatch_name)(
                input_a,
                input_b,
                out,
                ragged_tokens_info,
                0,
                resolve_num_mps(input_a.device, num_mp),
            )
        else:
            ragged_moe_gemm_16bit_mubin(
                input_a,
                input_b,
                ragged_tokens_info,
                out,
                False,
                None,
                alignment_m,
                num_mp,
            )
    elif gemm_mode == "per_expert":
        if backend == "mutlass":
            dispatch_name, mod = get_deep_gemm_gemm_module(
                kind="bf16",
                gemm_type=GEMM_TYPE_M_GROUPED_CONTIGUOUS,
                config_m=input_a.shape[0],
                alignment_m=alignment_m,
            )
            mod.get_function(dispatch_name)(
                input_a,
                input_b,
                out,
                ragged_tokens_info,
                0,
                resolve_num_mps(input_a.device, num_mp),
            )
        else:
            m_grouped_contig_gemm_16bit_mubin(
                input_a,
                input_b,
                ragged_tokens_info,
                out,
                major_a_mode,
                major_b_mode,
                num_mp,
            )
    else:
        assert False, "Not supported gemm mode."

    return out


@mate_api
def masked_moe_gemm_16bit(
    a: torch.Tensor,
    b: torch.Tensor,
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    expect_tokens: Optional[int] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    backend: Optional[Literal["auto", "mubin", "mutlass"]] = "auto",
):
    """
    Perform 16-bit GEMM operation for MoE (Mixture of Experts) with masked tensor inputs.

    This function computes matrix multiplication between 16-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens, using a mask to indicate
    the actual number of tokens per expert.

    Parameters
    ----------
    a : Tensor
        Input tensor A with shape ``(num_expert, max_tokens, hidden_size)`` in fp16/bf16 format.
    b : Tensor
        Input tensor B with shape ``(num_expert, out_hidden_size, hidden_size)`` in fp16/bf16 format.
    masked_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of fp16 or bf16 type. If None, a new tensor will be created.
    expect_tokens : Optional[int]
        Expected number of tokens. If None, defaults to 0.
    enable_overlap : Optional[bool]
        Whether to enable Single-Batch Overlap (SBO). Default is False.
    signal : Optional[Tensor]
        Signal tensor with shape ``(num_expert * ceil_div(max_m, 64))`` for
        SBO. Required if enable_overlap is True. If None, a new tensor will be
        created if needed.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, int, int]]
        If ``enable_overlap`` is ``False``, returns result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        If ``enable_overlap`` is ``True``, returns a tuple containing:

            - result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``
            - signal tensor
            - block_m int
            - threshold int

    """

    if expect_tokens is None:
        expect_tokens = 0

    backend = cast(
        Literal["auto", "mubin", "mutlass"],
        resolve_backend(backend, supported=("mubin", "mutlass"), default="auto"),
    )

    if not enable_overlap:
        signal = None

    if enable_overlap and signal is None:
        tile_signal = 64
        expert_sz = a.size(0)
        max_m = a.size(1)

        # zero init is required
        signal = torch.zeros(
            expert_sz * ceil_div(max_m, tile_signal),
            dtype=torch.int32,
            device=a.device,
        )

    if backend == "mutlass":
        if enable_overlap:
            raise NotImplementedError(
                'backend="mutlass" does not support enable_overlap'
            )
        dispatch_name, mod = get_deep_gemm_gemm_module(
            kind="bf16",
            gemm_type=GEMM_TYPE_M_GROUPED_MASKED,
            config_m=expect_tokens,
        )
        mod.get_function(dispatch_name)(
            a,
            b,
            out,
            masked_tokens_info,
            int(expect_tokens),
            resolve_num_mps(a.device),
        )
        return out

    res = masked_moe_gemm_16bit_mubin(
        a,
        b,
        masked_tokens_info,
        out,
        expect_tokens,
        signal,
    )

    return (out, signal, res[0], res[1]) if enable_overlap else out


@mate_api
def ragged_m_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[
        Literal["per_token", "psum_expert", "per_expert"]
    ] = "per_token",
    major_a_mode: Optional[Literal["M", "K"]] = "K",
    major_b_mode: Optional[Literal["N", "K"]] = "K",
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    num_mp: Optional[int] = None,
    alignment_m: Optional[int] = None,
    backend: Optional[Literal["auto", "mubin", "mutlass"]] = "auto",
):
    """
    Perform 8-bit GEMM operation for MoE (Mixture of Experts) with ragged tensor inputs.

    This function computes matrix multiplication between 8-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens assigned to them.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input A.
        **fp8_tensor** has shape ``(total_tokens, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(total_tokens, hidden_size // scale_granularity_m)`` and should be of fp32 type.
    input_b : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input B.
        **fp8_tensor** has shape ``(num_expert, out_hidden_size, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(num_expert, out_hidden_size // scale_granularity_n, hidden_size // scale_granularity_k)`` and should be of fp32 type.
    ragged_tokens_info : Tensor
        Metadata tensor whose meaning depends on ``gemm_mode``.
        For ``per_token``, it has shape ``(total_tokens,)`` and stores the
        expert index for each token, with ``-1`` for unused positions.
        For ``psum_expert``, it has shape ``(num_expert,)`` and stores how many
        tokens the leading experts have in prefix-sum form.
        For ``per_expert``, it has shape ``(num_expert,)`` and stores the token
        count for each expert.
    out : Tensor
        Output tensor with shape ``(total_tokens, out_hidden_size)``.
    major_a_mode : Optional[str]
        Indicating major stride of A.
        Default to `K`.
    major_b_mode : Optional[str]
        Indicating major stride of B.
        Default to `K`.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for total_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Default is ``(1, 128, 128)``.
    alignment_m : Optional[int]
        Alignment requirement for total_tokens (m) dimension. Must be 128 or 256.
        Default is 128.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Tensor
        Result tensor with shape ``(total_tokens, out_hidden_size)`` containing the GEMM output in fp16 or bf16 data type.

    """

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    if alignment_m is None:
        alignment_m = 128

    backend = cast(
        Literal["auto", "mubin", "mutlass"],
        resolve_backend(backend, supported=("mubin", "mutlass"), default="auto"),
    )

    if gemm_mode == "per_token":
        if backend == "mutlass":
            a_fp8, scale_a = input_a
            b_fp8, scale_b = input_b
            dispatch_name, mod = get_deep_gemm_gemm_module(
                kind="fp8",
                gemm_type=GEMM_TYPE_M_GROUPED_CONTIGUOUS,
                config_m=a_fp8.shape[0],
                alignment_m=alignment_m,
            )
            mod.get_function(dispatch_name)(
                a_fp8,
                scale_a,
                b_fp8,
                scale_b,
                out,
                ragged_tokens_info,
                0,
                resolve_num_mps(a_fp8.device, num_mp),
            )
        else:
            ragged_moe_gemm_8bit_mubin(
                input_a,
                input_b,
                ragged_tokens_info,
                scale_granularity_mnk,
                out,
                alignment_m,
                num_mp,
            )
    elif gemm_mode == "per_expert":
        if backend == "mutlass":
            a_fp8, scale_a = input_a
            b_fp8, scale_b = input_b
            dispatch_name, mod = get_deep_gemm_gemm_module(
                kind="fp8",
                gemm_type=GEMM_TYPE_M_GROUPED_CONTIGUOUS,
                config_m=a_fp8.shape[0],
                alignment_m=alignment_m,
            )
            mod.get_function(dispatch_name)(
                a_fp8,
                scale_a,
                b_fp8,
                scale_b,
                out,
                ragged_tokens_info,
                0,
                resolve_num_mps(a_fp8.device, num_mp),
            )
        else:
            m_grouped_contig_gemm_8bit_mubin(
                input_a,
                input_b,
                ragged_tokens_info,
                scale_granularity_mnk,
                out,
                major_a_mode,
                major_b_mode,
                num_mp,
            )
    else:
        assert False, "Not supported gemm mode"

    return out


@mate_api
def masked_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    expect_tokens: Optional[int] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    backend: Optional[Literal["auto", "mubin", "mutlass"]] = "auto",
):
    """
    Perform 8-bit GEMM operation for MoE (Mixture of Experts) with masked tensor inputs.

    This function computes matrix multiplication between 8-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens, using a mask to indicate
    the actual number of tokens per expert.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input A.
        **fp8_tensor** has shape ``(num_expert, max_tokens, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(num_expert, max_tokens, hidden_size // scale_granularity_k)`` and should be of fp32 type.
    input_b : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input B.
        **fp8_tensor** has shape ``(num_expert, out_hidden_size, hidden_size)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(num_expert, out_hidden_size // scale_granularity_n, hidden_size // scale_granularity_k)`` and should be of fp32 type.
    masked_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of fp16 or bf16 type. If None, a new tensor will be created.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for max_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Default is ``(1, 128, 128)``.
    expect_tokens : Optional[int]
        Expected number of tokens. If None, defaults to 0.
    enable_overlap : Optional[bool]
        Whether to enable Single-Batch Overlap (SBO). Default is False.
    signal : Optional[Tensor]
        Signal tensor with shape ``(num_expert * ceil_div(max_m, 64))`` for
        SBO. Required if ``enable_overlap`` is ``True``. If ``None``, a new
        tensor is created when needed.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, int, int]]
        If ``enable_overlap`` is ``False``, returns result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        If ``enable_overlap`` is ``True``, returns a tuple containing:

            - result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``
            - signal tensor
            - block_m int
            - threshold int

    """
    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    backend = cast(
        Literal["auto", "mubin", "mutlass"],
        resolve_backend(backend, supported=("mubin", "mutlass"), default="auto"),
    )

    if expect_tokens is None:
        expect_tokens = 0

    if not enable_overlap:
        signal = None

    if enable_overlap and signal is None:
        tile_signal = 64
        a, _ = input_a
        expert_sz = a.size(0)
        max_m = a.size(1)

        # zero init is required
        signal = torch.zeros(
            expert_sz * ceil_div(max_m, tile_signal),
            dtype=torch.int32,
            device=a.device,
        )

    if backend == "mutlass":
        if enable_overlap:
            raise NotImplementedError(
                'backend="mutlass" does not support enable_overlap'
            )
        a_fp8, scale_a = input_a
        b_fp8, scale_b = input_b
        dispatch_name, mod = get_deep_gemm_gemm_module(
            kind="fp8",
            gemm_type=GEMM_TYPE_M_GROUPED_MASKED,
            config_m=expect_tokens,
        )
        mod.get_function(dispatch_name)(
            a_fp8,
            scale_a,
            b_fp8,
            scale_b,
            out,
            masked_tokens_info,
            int(expect_tokens),
            resolve_num_mps(a_fp8.device),
        )
        return out

    res = masked_moe_gemm_8bit_mubin(
        input_a,
        input_b,
        masked_tokens_info,
        scale_granularity_mnk,
        out,
        expect_tokens,
        signal,
    )

    return (out, signal, res[0], res[1]) if enable_overlap else out


@mate_api
def ragged_moe_gemm_mixed_dtype(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    alignment_m: Optional[int] = None,
    *,
    mixed_dtype: GemmMixedDType | str,
    backend: Optional[Literal["auto", "mubin"]] = "auto",
    a_quant_recipe: Tuple[int, int],
    b_quant_recipe: Tuple[int, int],
):
    """
    Perform mixed-dtype GEMM operation for MoE (Mixture of Experts) with ragged tensor inputs.

    This function computes matrix multiplication between mixed-dtype tensors for MoE models
    where different experts may have variable numbers of tokens assigned to them.
    ``GemmMixedDType.S4FP8`` selects signed int4 weights, while
    ``GemmMixedDType.FP4FP8`` selects E2M1 FP4 weights. Both use FP8
    activations for input A.
    The quantization recipes describe quantization block sizes for input A and
    input B separately, and must be provided explicitly.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (activation_tensor, scale_tensor) for input A.
        **activation_tensor** has shape ``(total_tokens, hidden_size)``. Its dtype
        is selected by ``mixed_dtype``. S4FP8 accepts E4M3 or E5M2; FP4FP8
        requires E4M3.
        **scale_tensor** shape and dtype are selected by ``a_quant_recipe``. For
        ``a_quant_recipe=(1, -1)``, it has shape ``(total_tokens, 1)``.
    input_b : Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]
        For S4FP8, scales is one BF16 tensor with shape ``(num_expert,
        out_hidden_size, ceil_div(hidden_size, 128))``. For FP4FP8, scales is
        ``(residual_e8m0, epilogue_fp32)`` with shapes ``(num_expert,
        out_hidden_size, ceil_div(hidden_size, 32))`` and ``(num_expert,
        out_hidden_size)``.
    ragged_tokens_info : Tensor
        Tensor indicating which expert each token belongs to, with shape ``(total_tokens,)``.
        Values represent expert indices, with ``-1`` for unused positions.
    out : Tensor
        Output tensor with shape ``(total_tokens, out_hidden_size)``.
        S4FP8 supports FP16 or BF16; FP4FP8 requires BF16.
    alignment_m : Optional[int]
        S4FP8 accepts 128 or 256 and defaults to 128. FP4FP8 requires 256 and
        defaults to 256.
    mixed_dtype : GemmMixedDType or str
        Mixed dtype selector for input A and input B. Must be provided explicitly.
        ``GemmMixedDType.S4FP8`` and ``"s4fp8"`` mean signed int4 weights for
        input B and fp8 activations for input A.
        ``GemmMixedDType.FP4FP8`` and ``"fp4fp8"`` mean E2M1 FP4 weights and
        E4M3 activations.
    backend : Optional[str]
        Backend selector. Only ``"auto"`` and ``"mubin"`` are supported.
    a_quant_recipe : Tuple[int, int]
        Quantization block-size recipe for input A. The tuple is interpreted as
        ``(m, k)``. ``-1`` means the corresponding axis is not split into
        smaller quantization blocks. Currently, only ``(1, -1)`` is supported.
    b_quant_recipe : Tuple[int, int]
        Quantization block-size recipe for input B. The tuple is interpreted as
        ``(n, k)``. ``-1`` means the corresponding axis is not split into
        smaller quantization blocks. S4FP8 uses ``(1, 128)`` and FP4FP8 uses
        ``(1, 32)``.

    Returns
    -------
    Tensor
        Result tensor with shape ``(total_tokens, out_hidden_size)`` containing the GEMM output in fp16 or bf16 data type.

    """
    backend = cast(
        Literal["auto", "mubin"],
        resolve_backend(backend, supported=("mubin",), default="auto"),
    )
    if backend == "auto":
        backend = "mubin"
    mixed_dtype = _resolve_gemm_mixed_dtype(mixed_dtype)
    if alignment_m is None:
        alignment_m = 256 if mixed_dtype == GemmMixedDType.FP4FP8 else 128
    a_quant_recipe = _resolve_moe_gemm_quant_recipe("a_quant_recipe", a_quant_recipe)
    b_quant_recipe = _resolve_moe_gemm_quant_recipe("b_quant_recipe", b_quant_recipe)
    expected_b_recipe = (1, 32) if mixed_dtype == GemmMixedDType.FP4FP8 else (1, 128)
    valid_alignments = (256,) if mixed_dtype == GemmMixedDType.FP4FP8 else (128, 256)
    if (
        a_quant_recipe != (1, -1)
        or b_quant_recipe != expected_b_recipe
        or backend != "mubin"
        or alignment_m not in valid_alignments
    ):
        raise NotImplementedError(
            f"mixed_dtype={mixed_dtype.value}, a_quant_recipe={a_quant_recipe}, "
            f"b_quant_recipe={b_quant_recipe}, backend={backend} is not supported"
        )

    ragged_moe_gemm_w4a8_mubin(
        input_a,
        input_b,
        ragged_tokens_info,
        out,
        alignment_m,
        a_quant_recipe,
        b_quant_recipe,
        mixed_dtype=mixed_dtype.value,
    )
    return out


@mate_api
def masked_moe_gemm_mixed_dtype(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[
        torch.Tensor,
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    ],
    masked_tokens_info: torch.Tensor,
    out: torch.Tensor,
    expect_tokens: Optional[int] = None,
    enable_overlap: bool = False,
    signal: Optional[torch.Tensor] = None,
    *,
    mixed_dtype: GemmMixedDType | str,
    backend: Optional[Literal["auto", "mubin", "mutlass"]] = "auto",
    a_quant_recipe: Tuple[int, int],
    b_quant_recipe: Tuple[int, int],
):
    """
    Perform mixed-dtype GEMM operation for MoE (Mixture of Experts) with masked tensor inputs.

    This function computes matrix multiplication between mixed-dtype tensors for MoE models
    where different experts may have variable numbers of tokens, using a mask to indicate
    the actual number of tokens per expert. ``GemmMixedDType.S4FP8`` selects
    signed int4 weights, while ``GemmMixedDType.FP4FP8`` selects E2M1 FP4
    weights.
    The quantization recipes describe quantization block sizes for input A and
    input B separately, and must be provided explicitly.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (activation_tensor, scale_tensor) for input A.
        **activation_tensor** has shape ``(num_expert, max_tokens, hidden_size)``.
        Its dtype is selected by ``mixed_dtype``. S4FP8 accepts E4M3 or E5M2;
        FP4FP8 requires E4M3.
        **scale_tensor** shape and dtype are selected by ``a_quant_recipe``. For
        ``a_quant_recipe=(1, -1)``, it has shape ``(num_expert, max_tokens, 1)``.
        For ``a_quant_recipe=(1, 128)``, it has shape ``(num_expert,
        max_tokens, ceil_div(hidden_size, 128))``.
    input_b : Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]
        For S4FP8, scales is one BF16 tensor with shape ``(num_expert,
        out_hidden_size, ceil_div(hidden_size, 128))``. For FP4FP8, scales is
        ``(residual_e8m0, epilogue_fp32)`` with shapes ``(num_expert,
        out_hidden_size, ceil_div(hidden_size, 32))`` and ``(num_expert,
        out_hidden_size)``.
    masked_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        S4FP8 supports FP16 or BF16; FP4FP8 requires BF16.
    expect_tokens : Optional[int]
        Expected typical number of tokens per expert. A positive value participates
        in automatic backend selection. If None or 0, the tensor capacity is used.
    enable_overlap : Optional[bool]
        Whether to enable Single-Batch Overlap (SBO). Default is False.
    signal : Optional[Tensor]
        Signal tensor with shape ``(num_expert * ceil_div(max_m, 64))`` for
        SBO. Required if ``enable_overlap`` is ``True``. If ``None``, a new
        tensor is created when needed.
    mixed_dtype : GemmMixedDType or str
        Mixed dtype selector for input A and input B. Must be provided explicitly.
        ``GemmMixedDType.S4FP8`` and ``"s4fp8"`` mean signed int4 weights for
        input B and fp8 activations for input A.
        ``GemmMixedDType.FP4FP8`` and ``"fp4fp8"`` mean E2M1 FP4 weights and
        E4M3 activations.
    backend : Optional[str]
        Backend selector. ``"mutlass"`` uses the JIT/AOT MP31 kernel.
        ``"auto"`` selects it for compatible non-overlap inputs with
        ``min(max_tokens, expect_tokens) <= 32`` when ``expect_tokens`` is
        positive, and otherwise uses ``max_tokens`` as the threshold input.
        Grouped-A ``a_quant_recipe=(1, 128)`` always selects MUTLASS when its
        tensor contract is compatible because MUBIN grouped-A support is not
        available.
    a_quant_recipe : Tuple[int, int]
        Quantization block-size recipe for input A. The tuple is interpreted as
        ``(m, k)``. ``-1`` means the corresponding axis is not split into
        smaller quantization blocks. ``(1, -1)`` and ``(1, 128)`` are supported
        by the MUTLASS backend for ``GemmMixedDType.S4FP8``. MUBIN supports only
        ``(1, -1)``. Grouped Scale-A may be contiguous in either its M or K-block
        dimension. FP4FP8 supports only ``(1, -1)``.
    b_quant_recipe : Tuple[int, int]
        Quantization block-size recipe for input B. The tuple is interpreted as
        ``(n, k)``. ``-1`` means the corresponding axis is not split into
        smaller quantization blocks. S4FP8 uses ``(1, 128)`` and FP4FP8 uses
        ``(1, 32)``.

    Returns
    -------
    Union[Tensor, Tuple[Tensor, Tensor, int, int]]
        If ``enable_overlap`` is ``False``, returns result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        If ``enable_overlap`` is ``True``, returns a tuple containing:

            - result tensor with shape ``(num_expert, max_tokens, out_hidden_size)``
            - signal tensor
            - block_m int
            - threshold int

    """
    if expect_tokens is None:
        expect_tokens = 0

    backend = cast(
        Literal["auto", "mubin", "mutlass"],
        resolve_backend(backend, supported=("mubin", "mutlass"), default="auto"),
    )
    mixed_dtype = _resolve_gemm_mixed_dtype(mixed_dtype)
    a_quant_recipe = _resolve_moe_gemm_quant_recipe("a_quant_recipe", a_quant_recipe)
    b_quant_recipe = _resolve_moe_gemm_quant_recipe("b_quant_recipe", b_quant_recipe)
    if mixed_dtype == GemmMixedDType.FP4FP8:
        if (
            a_quant_recipe != (1, -1)
            or b_quant_recipe != (1, 32)
            or backend == "mutlass"
        ):
            raise NotImplementedError(
                f"mixed_dtype={mixed_dtype.value}, a_quant_recipe={a_quant_recipe}, "
                f"b_quant_recipe={b_quant_recipe}, backend={backend} is not supported"
            )
        backend = "mubin"
    elif not (
        mixed_dtype == GemmMixedDType.S4FP8
        and a_quant_recipe in _W4A8_MUTLASS_A_QUANT_RECIPES
        and b_quant_recipe == (1, 128)
    ):
        raise NotImplementedError(
            f"mixed_dtype={mixed_dtype.value}, a_quant_recipe={a_quant_recipe}, "
            f"b_quant_recipe={b_quant_recipe}, backend={backend} is not supported"
        )

    mutlass_input_b = cast(Tuple[torch.Tensor, torch.Tensor], input_b)
    if backend != "mubin":
        mutlass_enabled = check_w4a8_mutlass(
            input_a,
            mutlass_input_b,
            masked_tokens_info,
            out,
            a_quant_recipe,
            enable_overlap,
        )
        if backend == "auto":
            if a_quant_recipe == (1, 128):
                if not mutlass_enabled:
                    raise NotImplementedError(
                        'backend="mutlass" requires non-overlap E4M3 A with FP32 scales matching a_quant_recipe, '
                        "packed INT4/INT8 B with BF16 scales, FP16/BF16 output, "
                        "supported Scale-A major order, and compatible contiguous tensor layouts"
                    )
                backend = "mutlass"
            elif mutlass_enabled:
                max_m = int(input_a[0].shape[1])
                dispatch_m = min(max_m, expect_tokens) if expect_tokens > 0 else max_m
                backend = "mutlass" if dispatch_m <= 32 else "mubin"
            else:
                backend = "mubin"
        elif not mutlass_enabled:
            raise NotImplementedError(
                'backend="mutlass" requires non-overlap E4M3 A with FP32 scales matching a_quant_recipe, '
                "packed INT4/INT8 B with BF16 scales, FP16/BF16 output, "
                "supported Scale-A major order, and compatible contiguous tensor layouts"
            )

    if backend == "mutlass":
        return masked_moe_gemm_mixed_dtype_mutlass(
            input_a,
            mutlass_input_b,
            masked_tokens_info,
            out,
            expect_tokens,
            a_quant_recipe,
        )

    if a_quant_recipe != (1, -1):
        raise NotImplementedError(
            f'a_quant_recipe={a_quant_recipe}, backend="mubin" is not supported'
        )

    if not enable_overlap:
        signal = None

    if enable_overlap and signal is None:
        tile_signal = 64
        a, _ = input_a
        expert_sz = a.size(0)
        max_m = a.size(1)
        signal = torch.zeros(
            expert_sz * ceil_div(max_m, tile_signal),
            dtype=torch.int32,
            device=a.device,
        )

    res = masked_moe_gemm_w4a8_mubin(
        input_a,
        input_b,
        masked_tokens_info,
        out,
        expect_tokens,
        signal,
        a_quant_recipe,
        b_quant_recipe,
        mixed_dtype=mixed_dtype.value,
    )

    return (out, signal, res[0], res[1]) if enable_overlap else out


@mate_api
def ragged_k_moe_gemm_8bit(
    input_a: Tuple[torch.Tensor, torch.Tensor],
    input_b: Tuple[torch.Tensor, torch.Tensor],
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[Literal["per_expert"]] = "per_expert",
    major_a_mode: Optional[Literal["M", "K"]] = "M",
    major_b_mode: Optional[Literal["N", "K"]] = "N",
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    num_mp: Optional[int] = None,
):
    """
    Perform 8-bit GEMM operation for MoE (Mixture of Experts) with token of each expert.

    This function computes matrix multiplication between 8-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens.

    Parameters
    ----------
    input_a : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input A.
        **fp8_tensor** has shape ``(k, m)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(k // scale_granularity_k, m)`` and should be of fp32 type.
    input_b : Tuple[Tensor, Tensor]
        Tuple containing (fp8_tensor, scale_tensor) for input B.
        **fp8_tensor** has shape ``(k, n)`` and should be of fp8 (e4m3/e5m2) type.
        **scale_tensor** has shape ``(k // scale_granularity_k, n)`` and should be of fp32 type.
    ragged_tokens_info : Tensor
        Tensor indicating the actual number of tokens for each expert, with shape ``(num_expert,)``.
        Values represent token counts for each expert.
    out : Tensor
        Output tensor with shape ``(num_expert, max_tokens, out_hidden_size)``.
        Should be of float type. Should not be None.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    major_a_mode : Optional[str]
        Major mode of A, defult to `M`.
        Only support TN m_grouped_gemm on MP31.
    major_b_mode : Optional[str]
        Major mode of B, defult to `N`.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Quantization granularity for max_tokens, out_hidden_size, hidden_size (m, n, k) dimensions respectively.
        Kgroupgemm only support 1D1D scale, should be ``(1, 1, 128)``.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Result tensor with shape ``(num_experts, total_tokens, out_hidden_size)`` containing the GEMM output in float data type,
    Representing D = D + A * B for each expert

    """
    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 1, 128)
    else:
        assert scale_granularity_mnk == (1, 1, 128), (
            "k_grouped_contig_gemm_8bit only support 1D1D gemm"
        )

    if major_a_mode is None:
        major_a_mode = "M"
    if major_b_mode is None:
        major_b_mode = "N"

    assert major_a_mode == "M" and major_b_mode == "N", (
        "k_grouped_contig_gemm_8bit only support TN layout"
    )

    ragged_k_moe_gemm_8bit_mubin(
        input_a,
        input_b,
        ragged_tokens_info,
        scale_granularity_mnk,
        out,
        num_mp,
    )

    return out


@mate_api
def ragged_k_moe_gemm_16bit(
    input_a: torch.Tensor,
    input_b: torch.Tensor,
    ragged_tokens_info: torch.Tensor,
    out: torch.Tensor,
    gemm_mode: Optional[Literal["per_expert"]] = "per_expert",
    major_a_mode: Optional[Literal["M", "K"]] = "M",
    major_b_mode: Optional[Literal["N", "K"]] = "N",
    num_mp: Optional[int] = None,
):
    """
    Perform 16-bit GEMM operation for MoE (Mixture of Experts) with token of each expert.

    This function computes matrix multiplication between 16-bit quantized tensors for MoE models
    where different experts may have variable numbers of tokens.

    Parameters
    ----------
    input_a : Tensor
        Input tensor A with shape ``(sum(ks), m)`` in FP16 or BF16 format.
    input_b : Tensor
        Input tensor B with shape ``(sum(ks), n)`` and the same dtype as A.
    ragged_tokens_info : Tensor
        Per-expert K lengths with shape ``(num_expert,)``.
    out : Tensor
        Output tensor with shape ``(num_expert, m, n)``. FP32 is supported for
        FP16/BF16 inputs; BF16 output requires BF16 inputs.
    gemm_mode : Optional[str],
        Indicating different meaning of ragged_tokens_info.
    major_a_mode : Optional[str]
        Major mode of A, defult to `M`.
        Only support TN m_grouped_gemm on MP31.
    major_b_mode : Optional[str]
        Major mode of B, defult to `N`.
    num_mp : Optional[int]
        Suggest mp number.
        If None, will be get from device info.

    Returns
    -------
    Result tensor with shape ``(num_expert, m, n)`` containing the GEMM output in FP32 or BF16,
    Representing D = D + A * B for each expert

    """

    if major_a_mode is None:
        major_a_mode = "M"
    if major_b_mode is None:
        major_b_mode = "N"

    assert major_a_mode == "M" and major_b_mode == "N", (
        "k_grouped_contig_gemm_16bit only supports TN layout"
    )

    ragged_k_moe_gemm_16bit_mubin(
        input_a,
        input_b,
        ragged_tokens_info,
        out,
        num_mp,
    )

    return out


@functools.cache
def _get_bmm_module():
    return get_gemm_ops_module()


def _run_bmm_mubin_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    recipe_a: Tuple[int, int],
    recipe_b: Tuple[int, int],
    c: Optional[torch.Tensor],
    trans_a: bool,
    trans_b: bool,
    fixed_scale_layout: Optional[bool],
) -> None:
    if c is not None:
        raise ValueError('backend="mubin" does not support C accumulation')
    if recipe_a[1] != recipe_b[1]:
        raise ValueError("recipe_a and recipe_b must use matching K granularity")
    scale_granularity_mnk = (recipe_a[0], recipe_b[0], recipe_a[1])
    if fixed_scale_layout not in (None, False):
        raise ValueError('backend="mubin" only supports K-major scales')
    if fixed_scale_layout is False and (trans_a or not trans_b):
        raise ValueError(
            'backend="mubin" only supports explicit K-major scales for NT BMM'
        )

    batch = a.size(0)
    batch_tensors = {
        "b": b,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out": out,
        "scale_out": scale_out,
    }
    for name, tensor in batch_tensors.items():
        if tensor.size(0) != batch:
            raise ValueError(
                f"{name} batch dimension must be {batch}, got {tensor.size(0)}"
            )

    mubin_major_a = "MN" if trans_a else "K"
    mubin_major_b = "K" if trans_b else "MN"
    for batch_index in range(batch):
        groupwise_gemm_8bit_fp8output_mubin(
            (a[batch_index], scale_a[batch_index]),
            (b[batch_index], scale_b[batch_index]),
            scale_granularity_mnk,
            out[batch_index],
            scale_out[batch_index],
            mubin_major_a,
            mubin_major_b,
            None,
        )


def _run_bmm_mudnn(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    c: Optional[torch.Tensor],
    scale_a: Optional[torch.Tensor],
    scale_b: Optional[torch.Tensor],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
    trans_a: bool,
    trans_b: bool,
    fixed_scale_layout: Optional[bool],
) -> None:
    is_fp8 = a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8 and (recipe_a is None or recipe_b is None):
        raise ValueError("FP8 inputs require recipe_a and recipe_b")
    if is_fp8 and fixed_scale_layout is None:
        fixed_scale_layout = trans_a or not trans_b
    elif not is_fp8:
        scale_a = scale_b = None
        recipe_a = recipe_b = (-1, -1)
        fixed_scale_layout = False
    _get_bmm_module().get_function("bmm")(
        a,
        b,
        out,
        c,
        scale_a,
        scale_b,
        recipe_a,
        recipe_b,
        trans_a,
        trans_b,
        fixed_scale_layout,
    )


def _run_bmm_mutlass(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    c: Optional[torch.Tensor],
    scale_a: Optional[torch.Tensor],
    scale_b: Optional[torch.Tensor],
    recipe_a: Optional[Tuple[int, int]],
    recipe_b: Optional[Tuple[int, int]],
    trans_a: bool,
    trans_b: bool,
    fixed_scale_layout: Optional[bool],
) -> None:
    if trans_a or not trans_b:
        raise ValueError('backend="mutlass" only supports NT BMM')
    if c is not None:
        raise ValueError('backend="mutlass" does not support C accumulation')

    if a.dtype == torch.bfloat16:
        if b.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
            raise ValueError('backend="mutlass" requires BF16 A, B, and output')
        kind = "bf16"
    elif a.dtype == torch.float8_e4m3fn:
        if b.dtype != torch.float8_e4m3fn or out.dtype != torch.bfloat16:
            raise ValueError(
                'backend="mutlass" requires FP8 E4M3 A and B with BF16 output'
            )
        if scale_a is None or scale_b is None:
            raise ValueError('backend="mutlass" requires scale_a and scale_b')
        if recipe_a != (1, 128) or recipe_b != (128, 128):
            raise ValueError(
                'backend="mutlass" requires recipe_a=(1, 128) and recipe_b=(128, 128)'
            )
        if fixed_scale_layout not in (None, False):
            raise ValueError('backend="mutlass" only supports K-major scales')
        kind = "fp8"
    else:
        raise ValueError('backend="mutlass" only supports BF16 or FP8 E4M3 inputs')

    dispatch_name, mod = get_deep_gemm_gemm_module(
        kind=kind,
        gemm_type=GEMM_TYPE_NORMAL,
        config_m=a.size(1),
    )
    func = mod.get_function(dispatch_name)
    num_mps = resolve_num_mps(a.device)
    for batch_index in range(a.size(0)):
        if kind == "bf16":
            func(a[batch_index], b[batch_index], out[batch_index], None, 0, num_mps)
        else:
            assert scale_a is not None and scale_b is not None
            func(
                a[batch_index],
                scale_a[batch_index],
                b[batch_index],
                scale_b[batch_index],
                out[batch_index],
                None,
                0,
                num_mps,
            )


@mate_api
def bmm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    trans_a: bool = False,
    trans_b: bool = True,
    scale_a: Optional[torch.Tensor] = None,
    scale_b: Optional[torch.Tensor] = None,
    scale_out: Optional[torch.Tensor] = None,
    recipe_a: Optional[Tuple[int, int]] = None,
    recipe_b: Optional[Tuple[int, int]] = None,
    c: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    fixed_scale_layout: Optional[bool] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Perform FP8, FP16, or BF16 batched matrix multiplication.

    This function computes the batched matrix product of A and B, optionally
    adds C, and stores the result in the requested output dtype. It supports
    unscaled FP16/BF16 inputs and FP8 inputs with tensorwise, channelwise, or
    groupwise scaling.

    Parameters
    ----------
    a : torch.Tensor
        Input A. Its physical shape is ``(batch, m, k)`` when
        ``trans_a=False`` or ``(batch, k, m)`` when ``trans_a=True``.
        Supported dtypes are FP16, BF16, FP8 E4M3, and FP8 E5M2. The final
        physical dimension must have stride 1.
    b : torch.Tensor
        Input B. Its physical shape is ``(batch, n, k)`` when
        ``trans_b=True`` or ``(batch, k, n)`` when ``trans_b=False``.
        It must use the same 16-bit dtype as A for unscaled BMM, or an FP8
        dtype for scaled BMM. The final physical dimension must have stride 1.
    out : Optional[torch.Tensor]
        Preallocated output D with shape ``(batch, m, n)``. When omitted, a
        tensor is allocated using ``out_dtype``. The default dtype is the input
        dtype for 16-bit BMM and BF16 for FP8 BMM without ``scale_out``.
    trans_a : bool
        Whether to transpose the final two dimensions of physical A before
        multiplication. Default is False.
    trans_b : bool
        Whether to transpose the final two dimensions of physical B before
        multiplication. Default is True.
    scale_a : Optional[torch.Tensor]
        FP32 scaling factors for A. Required for FP8 inputs and ignored for
        16-bit inputs. Its logical granularity is specified by ``recipe_a``.
    scale_b : Optional[torch.Tensor]
        FP32 scaling factors for B. Required for FP8 inputs and ignored for
        16-bit inputs. Its logical granularity is specified by ``recipe_b``.
    scale_out : Optional[torch.Tensor]
        FP32 output scales for FP8 E4M3 output. Providing this tensor selects
        the MUBIN backend; its shape is ``(batch, m, ceil(n / 128))``.
        Ignored for 16-bit inputs.
    recipe_a : Optional[Tuple[int, int]]
        Required FP8 quantization recipe ``(m_granularity, k_granularity)``
        for A. ``(-1, -1)``, ``(1, -1)``, and ``(1, 128)`` represent
        tensorwise, channelwise, and K-grouped scaling, respectively.
    recipe_b : Optional[Tuple[int, int]]
        Required FP8 quantization recipe ``(n_granularity, k_granularity)``
        for B. In addition to tensorwise, channelwise, and grouped scaling,
        ``(128, 128)`` represents block scaling. Its K granularity must match
        ``recipe_a``.
    c : Optional[torch.Tensor]
        Optional accumulation tensor with shape ``(batch, m, n)``. It must
        match the output dtype. FP8 BMM with C requires FP32 output. The MUBIN
        and MUTLASS backends do not support C.
    out_dtype : Optional[torch.dtype]
        Output dtype used only when ``out`` is omitted. The muDNN backend
        accepts the 16-bit input dtype or FP32 for unscaled BMM, and FP16,
        BF16, or FP32 for FP8 BMM. MUBIN requires FP8 E4M3 output. MUTLASS
        requires BF16 output.
    fixed_scale_layout : Optional[bool]
        Common packed layout for non-scalar FP8 scales. False selects K-major,
        True selects MN-major, and None selects K-major for NT or MN-major for
        NN, TN, and TT. Ignored for 16-bit inputs.
    backend : str
        Backend selector. ``"auto"`` uses muDNN unless ``scale_out`` selects
        MUBIN. Explicitly supported backends are ``"mudnn"``, ``"mubin"``, and
        ``"mutlass"``. MUTLASS supports NT BF16 or group/block FP8 E4M3 BMM
        with BF16 output.

    Returns
    -------
    torch.Tensor
        Output D with shape ``(batch, m, n)``. If ``out`` is provided, the same
        tensor is returned.
    """
    is_fp8 = a.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and b.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if is_fp8:
        if scale_a is None or scale_b is None:
            raise ValueError("FP8 inputs require scale_a and scale_b")
        if recipe_a is None or recipe_b is None:
            raise ValueError("FP8 inputs require recipe_a and recipe_b")
        group_or_block_scaled = (
            recipe_a[0] == 1
            and recipe_a[1] == 128
            and recipe_b[0] in (1, 128)
            and recipe_b[1] == 128
        )
        if (
            group_or_block_scaled
            and a.dtype == torch.float8_e4m3fn
            and b.dtype == torch.float8_e5m2
        ):
            raise ValueError(
                "FP8 bmm group/block scaling does not support E4M3 a with E5M2 b"
            )
        backend = resolve_backend(
            backend,
            supported=("mudnn", "mubin", "mutlass"),
            allow_auto=True,
            default="auto",
        )
        if scale_out is not None:
            if backend == "auto":
                backend = "mubin"
            elif backend != "mubin":
                raise ValueError("scale_out requires the mubin backend")
        elif backend == "mubin":
            raise ValueError('backend="mubin" requires scale_out')
        elif backend == "auto":
            backend = "mudnn"
    else:
        if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("unscaled bmm only supports FP16 or BF16 inputs")
        backend = resolve_backend(
            backend,
            supported=("mudnn", "mutlass"),
            allow_auto=True,
            default="auto",
        )
        if backend == "auto":
            backend = "mudnn"

    batch = a.size(0)
    m = a.size(2) if trans_a else a.size(1)
    n = b.size(1) if trans_b else b.size(2)
    if out is None:
        if out_dtype is None:
            out_dtype = (
                torch.float8_e4m3fn if is_fp8 and scale_out is not None else a.dtype
            )
        if (
            is_fp8
            and scale_out is None
            and out_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        ):
            out_dtype = torch.bfloat16
        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)

    if backend == "mudnn":
        _run_bmm_mudnn(
            a,
            b,
            out,
            c,
            scale_a,
            scale_b,
            recipe_a,
            recipe_b,
            trans_a,
            trans_b,
            fixed_scale_layout,
        )
    elif backend == "mutlass":
        _run_bmm_mutlass(
            a,
            b,
            out,
            c,
            scale_a,
            scale_b,
            recipe_a,
            recipe_b,
            trans_a,
            trans_b,
            fixed_scale_layout,
        )
    else:
        if (
            scale_a is None
            or scale_b is None
            or scale_out is None
            or recipe_a is None
            or recipe_b is None
        ):
            raise ValueError('backend="mubin" requires FP8 scales and scale_out')
        _run_bmm_mubin_fp8(
            a,
            b,
            scale_a,
            scale_b,
            out,
            scale_out,
            recipe_a,
            recipe_b,
            c,
            trans_a,
            trans_b,
            fixed_scale_layout,
        )
    return out
