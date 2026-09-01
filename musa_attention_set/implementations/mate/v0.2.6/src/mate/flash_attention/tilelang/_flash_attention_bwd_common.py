# ruff: noqa
# type: ignore
from dataclasses import replace
import torch
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import itertools
from ...utils import cosize

TARGET = "musa"
DEVICE = "musa"

INT32_ADDRESS_SPACE_BYTES = torch.iinfo(torch.int32).max


def _make_jit_pass_configs(disable_index_type_promotion=True):
    return {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: disable_index_type_promotion,
    }


JIT_PASS_CONFIGS = _make_jit_pass_configs(disable_index_type_promotion=True)
JIT_PASS_CONFIGS_PROMOTE_INDEX = _make_jit_pass_configs(
    disable_index_type_promotion=False
)

JIT_COMPILE_FLAGS = [
    # "-Od3",
    "-fmusa-flush-denormals-to-zero",
    "-fno-signed-zeros",
    "-fno-strict-aliasing",
    "-mllvm",
    "-misched=mtgpu-max-ilp",
    "-mllvm",
    "-mtgpu-tiny-offset-hint=1",
    "-mllvm",
    "-misched-recompute-slotindex=1",
    "-mllvm",
    "-mtgpu-if-convert=1",
    # "-mllvm",
    # "-mtgpu-combine-instr-with-burst=1",
    "-mllvm",
    "-mtgpu-combine-fop-instr=1",
]


def perm_n(n, block_N, atom_n=None):
    # assume n in blockN; optionally permute independently inside atom_n chunks
    if atom_n is None:
        atom_n = block_N
    local_n = n % atom_n
    atom_base = n // atom_n * atom_n
    l = atom_n // 8
    return atom_base + (local_n % 8) * l + (local_n // 8)


def _check_last_dim_stride_one(name, tensor):
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} must have contiguous last dimension")


def _check_attention_strides(name, tensor, multiple=8):
    _check_last_dim_stride_one(name, tensor)
    for dim, stride in enumerate(tensor.stride()[:-1]):
        if stride <= 0:
            raise ValueError(f"{name} stride({dim}) must be positive, got {stride}")
        if stride % multiple != 0:
            raise ValueError(
                f"{name} stride({dim}) must be divisible by {multiple}, got {stride}"
            )


def _tilelang_dtype_nbytes(dtype):
    if dtype in ("float16", "bfloat16"):
        return 2
    if dtype in ("float", "float32"):
        return 4
    raise ValueError(f"Unsupported TileLang dtype: {dtype}")


def _cosize_bytes(shape, strides, element_size):
    if any(extent == 0 for extent in shape):
        return 0
    return cosize(shape, strides) * element_size


def _tensor_cosize_bytes(tensor):
    return _cosize_bytes(
        tuple(tensor.shape), tuple(tensor.stride()), tensor.element_size()
    )


def _contiguous_cosize_bytes(shape, torch_dtype):
    return _cosize_bytes(
        tuple(shape), None, torch.empty((), dtype=torch_dtype).element_size()
    )


def _needs_index_type_promotion(*byte_spans):
    return any(byte_span > INT32_ADDRESS_SPACE_BYTES for byte_span in byte_spans)


def _clone_jit_with_pass_configs(jit_impl, pass_configs):
    return replace(jit_impl, pass_configs=dict(pass_configs))


_JIT_INDEX_PROMOTION_VARIANTS = {}


def _jit_for_index_type_promotion(jit_impl, enable_index_type_promotion):
    if not enable_index_type_promotion:
        return jit_impl
    promote = getattr(jit_impl, "with_index_type_promotion", None)
    if promote is not None:
        return promote()
    key = id(jit_impl)
    variant = _JIT_INDEX_PROMOTION_VARIANTS.get(key)
    if variant is None:
        variant = _clone_jit_with_pass_configs(jit_impl, JIT_PASS_CONFIGS_PROMOTE_INDEX)
        _JIT_INDEX_PROMOTION_VARIANTS[key] = variant
    return variant


def _annotate_sqmma(buffer, k_major, continuity=None):
    buffer_view = buffer if hasattr(buffer, "buffer") else buffer[:, :]
    if continuity is None:
        layout = tilelang.layout.make_sqmma_swizzled_layout(
            buffer_view, k_major=k_major
        )
    else:
        layout = tilelang.layout.make_sqmma_swizzled_layout(
            buffer_view, k_major=k_major, continuity=continuity
        )
    T.annotate_layout(
        {buffer_view: layout},
        allow_reannotation=True,
        allow_buffer_region=True,
    )
