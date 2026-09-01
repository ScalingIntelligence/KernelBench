import functools
from enum import Enum
from typing import Literal, Optional, Tuple, Union, cast

import torch

from mate.api_logging import mate_api
from mate._backend import resolve_backend
from mate.jit.gemm.deep_gemm.gemm import (
    GEMM_TYPE_M_GROUPED_CONTIGUOUS,
    GEMM_TYPE_M_GROUPED_MASKED,
    get_deep_gemm_gemm_module,
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


@functools.cache
def _get_module():
    return get_gemm_ops_module()


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
        Signal tensor with shape ``(num_expert * ceil_div(max_m, 64))``for SBO. Required if enable_overlap is True. If None, a new tensor will be created if needed.

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
    backend: Optional[Literal["auto", "mubin"]] = "auto",
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
    input_b : Tuple[Tensor, Union[Tensor, Tuple[Tensor, Tensor]]]
        For S4FP8, scales is one BF16 tensor. For FP4FP8, scales is
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
        Expected number of tokens. If None, defaults to 0.
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
        Literal["auto", "mubin"],
        resolve_backend(backend, supported=("mubin",), default="auto"),
    )
    if backend == "auto":
        backend = "mubin"
    mixed_dtype = _resolve_gemm_mixed_dtype(mixed_dtype)
    a_quant_recipe = _resolve_moe_gemm_quant_recipe("a_quant_recipe", a_quant_recipe)
    b_quant_recipe = _resolve_moe_gemm_quant_recipe("b_quant_recipe", b_quant_recipe)
    expected_b_recipe = (1, 32) if mixed_dtype == GemmMixedDType.FP4FP8 else (1, 128)
    if (
        a_quant_recipe != (1, -1)
        or b_quant_recipe != expected_b_recipe
        or backend != "mubin"
    ):
        raise NotImplementedError(
            f"mixed_dtype={mixed_dtype.value}, a_quant_recipe={a_quant_recipe}, "
            f"b_quant_recipe={b_quant_recipe}, backend={backend} is not supported"
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


@mate_api
def bmm_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    output_scale: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    major_a_mode: Literal["K", "M"] = "K",
    major_b_mode: Literal["N", "K"] = "K",
):
    """
    Perform batched matrix multiplication with FP8 quantized tensors.

    This function computes the batched matrix multiplication of two FP8 quantized tensors,
    applying scaling factors to produce a result in the specified output data type.

    Parameters
    ----------
    a : Tensor
        Input tensor A in FP8 format (e4m3/e5m2). Shape is
        ``(batch, m, k)`` when ``major_a_mode="K"`` and
        ``(batch, k, m)`` when ``major_a_mode="M"``. The declared major
        matrix dimension must have stride 1.
    b : Tensor
        Input tensor B in FP8 format (e4m3/e5m2). Shape is
        ``(batch, n, k)`` by default with ``major_b_mode="K"`` and
        ``(batch, k, n)`` when ``major_b_mode="N"``. The declared
        major matrix dimension must have stride 1.
    a_scale : Tensor
        Scaling factors for tensor A with shape depending on scale_granularity.
        Should be of fp32 type.
    b_scale : Tensor
        Scaling factors for tensor B with shape depending on scale_granularity.
        Should be of fp32 type.
    out_dtype : torch.dtype
        Data type for the output tensor. torch.bfloat16, torch.float16 and
        torch.float32 are supported.
    out : Optional[Tensor]
        Pre-allocated output tensor with shape ``(batch, m, n)``.
        Default is None.
        If None, a new tensor will be allocated.
    backend : str
        Backend to use for the operation.
        Current support backends are "mudnn" and "auto".
        Default is "auto".
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Granularity of scaling for batch, m, and n dimensions respectively.
        ``(-1, -1, -1)``, ``(1, -1, -1)``, ``(1, 128, 128)`` and
        ``(1, 1, 128)`` are supported.
        If None, defaults to ``(-1, -1, -1)``.
    c : Optional[Tensor]
        Optional FP32 accumulation tensor with shape ``(batch, m, n)``.
    major_a_mode : str
        ``"K"`` treats A as ``(batch, m, k)``; ``"M"`` treats A as
        ``(batch, k, m)`` and asks MatMulLt to transpose A.
    major_b_mode : str
        ``"K"`` treats B as ``(batch, n, k)`` and asks MatMulLt to
        transpose B. ``"N"`` treats B as ``(batch, k, n)``.
        Default is ``"K"``.

    Returns
    -------
    Tensor
        Result tensor with shape ``(batch, m, n)`` in the specified output data type.
    """
    backend = resolve_backend(
        backend, supported=("mudnn",), allow_auto=True, default="auto"
    )

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (-1, -1, -1)

    if major_a_mode not in ("K", "M"):
        raise ValueError("major_a_mode must be either 'K' or 'M'")
    if major_b_mode not in ("N", "K"):
        raise ValueError("major_b_mode must be either 'N' or 'K'")

    batch = a.size(0)
    if major_a_mode == "K":
        m = a.size(1)
        k = a.size(2)
    else:
        k = a.size(1)
        m = a.size(2)

    if major_b_mode == "N":
        b_k = b.size(1)
        n = b.size(2)
    else:
        n = b.size(1)
        b_k = b.size(2)

    if b.size(0) != batch or b_k != k:
        raise ValueError(
            "bmm_fp8 expects A as [batch,m,k] or [batch,k,m] according to "
            "major_a_mode, and B as [batch,k,n] or [batch,n,k] according to "
            "major_b_mode"
        )

    if out is None:
        if out_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError("Only bf16, fp16 and fp32 are supported for out_type!")

        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)
    else:
        if tuple(out.shape) != (batch, m, n):
            raise ValueError(
                f"out must have shape {(batch, m, n)}, got {tuple(out.shape)}"
            )
        if out.device != a.device:
            raise ValueError(f"out must be on device {a.device}, got {out.device}")
        if out.dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError("Only bf16, fp16 and fp32 are supported for out_type!")

    if c is not None:
        if tuple(c.shape) != (batch, m, n):
            raise ValueError(f"c must have shape {(batch, m, n)}, got {tuple(c.shape)}")
        if c.device != a.device:
            raise ValueError(f"c must be on device {a.device}, got {c.device}")
        if c.dtype != out.dtype:
            raise ValueError("c must have the same dtype as out")
        if out.dtype != torch.float32:
            raise ValueError("bmm_fp8 with c only supports fp32 output")

    if c is not None and k == 0:
        if out.data_ptr() != c.data_ptr():
            out.copy_(c)
        return out

    _get_module().get_function("bmm_fp8")(
        a,
        b,
        a_scale,
        b_scale,
        out,
        scale_granularity_mnk,
        backend,
        c,
        major_a_mode,
        major_b_mode,
    )
    return out


@mate_api
def bmm_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor] = None,
    backend: str = "auto",
    c: Optional[torch.Tensor] = None,
):
    backend = resolve_backend(
        backend, supported=("mudnn",), allow_auto=True, default="auto"
    )
    if out is None:
        batch = a.size(0)
        m = a.size(1)
        n = b.size(2)

        if out_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError("Only bf16, fp16 and fp32 are supported for out_type!")

        out = torch.empty((batch, m, n), dtype=out_dtype, device=a.device)

    if c is not None and a.size(2) == 0:
        if out.data_ptr() != c.data_ptr():
            out.copy_(c)
        return out

    _get_module().get_function("bmm_fp16")(a, b, out, c, backend)
    return out


@mate_api
def gemm_fp8_nt_groupwise(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[Literal["MN", "K"]] = None,
    mma_sm: Optional[int] = None,
    scale_granularity_mnk: Optional[Tuple[int, int, int]] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    backend: str = "auto",
    output_scale: Optional[torch.Tensor] = None,
):
    """
    Perform groupwise FP8 GEMM operation with scaling.

    This function computes the matrix multiplication of two FP8 quantized tensors, applying scaling factors to produce a result
    in the specified output data type. It supports groupwise quantization with configurable
    scale granularity.

    Parameters
    ----------
    a : Tensor
        Input tensor A with shape ``(m, k)`` in FP8 format (e4m3/e5m2).
        Tensor must be contiguous.
    b : Tensor
        Input tensor B with shape ``(n, k)`` in FP8 format (e4m3/e5m2).
        Tensor must be contiguous.
    a_scale : Tensor
        Scaling factors for tensor A. Shape depends on scale_granularity_mnk parameter.
        Should be of fp32 type. Must be contiguous.
    b_scale : Tensor
        Scaling factors for tensor B. Shape depends on scale_granularity_mnk parameter.
        Should be of fp32 type. Must be contiguous.
    scale_major_mode : str
        Scale major mode "MN" or "K" for groupwise operations. Default is "K".
    mma_sm : Optional[int]
        MMA SM configuration. Currently only supports 1. Default is 1.
    scale_granularity_mnk : Optional[Tuple[int, int, int]]
        Granularity of scaling for m, n, and k dimensions respectively.
        Default is ``(1, 128, 128)``.
    out : Optional[Tensor]
        Pre-allocated output tensor with shape ``(m, n)``.
        Should be bf16/fp16 when ``output_scale`` is None, or fp8_e4m3 when
        ``output_scale`` is provided. If None, a new tensor will be allocated.
    out_dtype : Optional[torch.dtype]
        Data type for the output tensor when ``out`` is None. If ``out`` is
        provided, ``out.dtype`` is validated instead.
        Defaults to torch.bfloat16 without ``output_scale`` and fp8_e4m3 with
        ``output_scale``.
    backend : str
        Backend to use for the operation. Use ``"mudnn"`` when
        ``output_scale`` is None and ``"mubin"`` when ``output_scale`` is
        provided. ``"auto"`` selects the supported backend for the selected
        output path.
    output_scale: Optional[torch.Tensor]
        Quantization scale tensor for FP8 output. If provided, the operation
        uses the mubin FP8-output path. If None, output is not quantized.
        Default is None.

    Returns
    -------
    Tensor
        Result tensor with shape ``(m, n)`` in the specified output data type.
    """

    if output_scale is None:
        backend = resolve_backend(
            backend, supported=("mudnn",), allow_auto=True, default="auto"
        )
        if backend == "auto":
            backend = "mudnn"
    else:
        # The FP8-output kernel is implemented by the mubin backend.
        backend = resolve_backend(
            backend, supported=("mubin",), allow_auto=True, default="auto"
        )
        if backend == "auto":
            backend = "mubin"

    if scale_major_mode is None:
        scale_major_mode = "K"

    if mma_sm is None:
        mma_sm = 1

    if mma_sm != 1:
        mma_sm = 1
        print("Warning: only mma_sm=1 is supported now, set mma_sm=1")

    if scale_granularity_mnk is None:
        scale_granularity_mnk = (1, 128, 128)

    major_a_mode = "K"
    major_b_mode = "K"

    m = a.size(0)
    n = b.size(0)
    expected_out_shape = (m, n)

    supported_out_dtypes: Tuple[torch.dtype, ...]
    if output_scale is not None:
        supported_out_dtypes = (torch.float8_e4m3fn,)
        default_out_dtype = torch.float8_e4m3fn
        out_dtype_error = "fp8_output only supports e4m3 now"
    else:
        supported_out_dtypes = (torch.bfloat16, torch.float16)
        default_out_dtype = torch.bfloat16
        out_dtype_error = "Only bf16 and fp16 are supported for out_type!"

    if out is None:
        if out_dtype is None:
            out_dtype = default_out_dtype

        if out_dtype not in supported_out_dtypes:
            raise ValueError(out_dtype_error)

        out = torch.empty(expected_out_shape, dtype=out_dtype, device=a.device)
    else:
        if not isinstance(out, torch.Tensor):
            raise TypeError("out must be a torch.Tensor")
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(
                f"out must have shape {expected_out_shape}, got {tuple(out.shape)}"
            )
        if out.device != a.device:
            raise ValueError(f"out must be on device {a.device}, got {out.device}")
        if out.dtype not in supported_out_dtypes:
            raise ValueError(out_dtype_error)
        if output_scale is None and out.stride(-1) != 1:
            raise ValueError("out must be contiguous at the last dimension")

    if output_scale is not None:
        groupwise_gemm_8bit_fp8output_mubin(
            (a, a_scale),
            (b, b_scale),
            scale_granularity_mnk,
            out,
            output_scale,
            major_a_mode,
            major_b_mode,
            None,
        )
    else:
        _get_module().get_function("gemm_fp8_nt_groupwise")(
            a,
            b,
            a_scale,
            b_scale,
            scale_major_mode,
            mma_sm,
            scale_granularity_mnk,
            out,
            backend,
        )
    return out
