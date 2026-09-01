import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

_M = 32
_N = 48
_BLOCK_M = 16
_BLOCK_N = 16


def _torch_dtype(tilelang_dtype):
    return {
        T.bool: torch.bool,
        T.float64: torch.float64,
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
        T.float8_e4m3fn: torch.float8_e4m3fn,
        T.int64: torch.int64,
        T.int32: torch.int32,
        T.int16: torch.int16,
        T.int8: torch.int8,
        T.uint64: torch.uint64,
        T.uint32: torch.uint32,
        T.uint16: torch.uint16,
        T.uint8: torch.uint8,
    }[tilelang_dtype]


def _make_input(shape, dtype):
    numel = shape[0] * shape[1]
    if dtype is T.bool:
        return (torch.arange(numel, dtype=torch.int32).reshape(shape) % 3) == 0
    if dtype in (T.float64, T.float32, T.float16, T.bfloat16, T.float8_e4m3fn):
        values = torch.arange(numel, dtype=torch.float32).reshape(shape)
        return (values / 13.0).to(_torch_dtype(dtype))
    if dtype in (T.int64, T.int32, T.int16):
        values = torch.arange(numel, dtype=torch.int64).reshape(shape) - 71
        return values.to(_torch_dtype(dtype))
    if dtype is T.int8:
        values = (torch.arange(numel, dtype=torch.int16).reshape(shape) % 127) - 63
        return values.to(torch.int8)
    if dtype in (T.uint64, T.uint32, T.uint16):
        values = torch.arange(numel, dtype=torch.int64).reshape(shape) % 1021
        return values.to(_torch_dtype(dtype))
    if dtype is T.uint8:
        values = torch.arange(numel, dtype=torch.int16).reshape(shape) % 251
        return values.to(torch.uint8)
    raise TypeError(f"unsupported dtype: {dtype}")


def _make_transpose_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_M, _N), dtype),
        dst: T.Tensor((_N, _M), dtype),
    ):
        with T.Kernel(T.ceildiv(_N, _BLOCK_N), T.ceildiv(_M, _BLOCK_M), threads=128) as (
            bx,
            by,
        ):
            tile = T.alloc_shared((_BLOCK_M, _BLOCK_N), dtype)
            tile_t = T.alloc_shared((_BLOCK_N, _BLOCK_M), dtype)

            T.copy(
                src[
                    by * _BLOCK_M : (by + 1) * _BLOCK_M,
                    bx * _BLOCK_N : (bx + 1) * _BLOCK_N,
                ],
                tile,
            )
            T.transpose(tile, tile_t)
            T.copy(
                tile_t,
                dst[
                    bx * _BLOCK_N : (bx + 1) * _BLOCK_N,
                    by * _BLOCK_M : (by + 1) * _BLOCK_M,
                ],
            )

    return kernel


@pytest.mark.parametrize(
    "dtype",
    [
        T.bool,
        T.float64,
        T.float32,
        T.float16,
        T.bfloat16,
        T.float8_e4m3fn,
        T.int64,
        T.int32,
        T.int16,
        T.int8,
        T.uint64,
        T.uint32,
        T.uint16,
        T.uint8,
    ],
)
@tilelang.testing.requires_musa
def test_musa_transpose_runtime_values(dtype):
    kernel = tilelang.compile(
        _make_transpose_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src = _make_input((_M, _N), dtype).to("musa")
    out = kernel(src)
    torch.musa.synchronize()

    torch.testing.assert_close(out.cpu(), src.cpu().T, atol=0, rtol=0)
