import re
from math import prod

import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


OOB_SHAPES = [
    ((20,), (32,)),
    ((4, 20), (4, 32)),
    ((2, 3, 20), (2, 4, 32)),
    ((2, 2, 3, 20), (2, 2, 4, 32)),
    ((2, 2, 2, 3, 20), (2, 2, 2, 4, 32)),
]


def _make_oob_tme_load(global_shape, shared_shape, dtype="float32"):
    global_shape = tuple(global_shape)
    shared_shape = tuple(shared_shape)
    starts = (0,) * len(global_shape)

    @T.prim_func
    def kernel(A: T.Tensor(global_shape, dtype), C: T.Tensor(shared_shape, dtype)):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared(shared_shape, dtype)
            barrier = T.alloc_barrier(128)
            # The logical source and destination tile extents stay equal. The
            # descriptor retains A's real shape, so hardware zero-fills box
            # positions that lie outside it.
            T.tma_copy(A[starts], smem, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            if len(shared_shape) == 1:
                for i0 in T.Parallel(shared_shape[0]):
                    C[i0] = smem[i0]
            elif len(shared_shape) == 2:
                for i0, i1 in T.Parallel(shared_shape[0], shared_shape[1]):
                    C[i0, i1] = smem[i0, i1]
            elif len(shared_shape) == 3:
                for i0, i1, i2 in T.Parallel(
                    shared_shape[0], shared_shape[1], shared_shape[2]
                ):
                    C[i0, i1, i2] = smem[i0, i1, i2]
            elif len(shared_shape) == 4:
                for i0, i1, i2, i3 in T.Parallel(
                    shared_shape[0],
                    shared_shape[1],
                    shared_shape[2],
                    shared_shape[3],
                ):
                    C[i0, i1, i2, i3] = smem[i0, i1, i2, i3]
            else:
                for i0, i1, i2, i3, i4 in T.Parallel(
                    shared_shape[0],
                    shared_shape[1],
                    shared_shape[2],
                    shared_shape[3],
                    shared_shape[4],
                ):
                    C[i0, i1, i2, i3, i4] = smem[i0, i1, i2, i3, i4]

    return kernel


def _make_tail_tme_load():
    @T.prim_func
    def kernel(A: T.Tensor((100,), "float32"), C: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            smem = T.alloc_shared((32,), "float32")
            barrier = T.alloc_barrier(32)
            T.tma_copy(A[80], smem, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            for i in T.Parallel(32):
                C[i] = smem[i]

    return kernel


def _lower(program):
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        return tilelang.lower(program, target=target, enable_device_compile=False)


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_oob_zero_fill_codegen():
    for global_shape, shared_shape in OOB_SHAPES:
        source = _lower(_make_oob_tme_load(global_shape, shared_shape)).kernel_source
        assert "tl::tme_load(" in source
        rank = len(shared_shape)
        call = next(line for line in source.splitlines() if "tl::tme_load(" in line)
        assert len(re.findall(r"\b32\b", call)) >= 1
        assert call.count(",") == 2 * rank + 2
        assert f"tl::tme_barrier_add_trans(1, {prod(shared_shape) * 4});" in source


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_oob_zero_fill_data_consistency():
    for global_shape, shared_shape in OOB_SHAPES:
        kernel = tilelang.compile(
            _make_oob_tme_load(global_shape, shared_shape),
            out_idx=[1],
            target={"kind": "musa", "arch": "mp_31"},
            execution_backend="tvm_ffi",
        )
        values = torch.randn(global_shape, dtype=torch.float32)
        result = kernel(values.to("musa"))
        torch.musa.synchronize()
        expected = torch.zeros(shared_shape, dtype=torch.float32)
        expected[tuple(slice(0, extent) for extent in global_shape)] = values
        torch.testing.assert_close(result.cpu(), expected)


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_oob_zero_fill_tail_data_consistency():
    kernel = tilelang.compile(
        _make_tail_tme_load(),
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    values = torch.randn((100,), dtype=torch.float32)
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    expected = torch.zeros((32,), dtype=torch.float32)
    expected[:20] = values[80:]
    torch.testing.assert_close(result.cpu(), expected)


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_oob_zero_fill_bfloat16_data_consistency():
    global_shape = (4, 72)
    shared_shape = (4, 128)
    kernel = tilelang.compile(
        _make_oob_tme_load(global_shape, shared_shape, dtype="bfloat16"),
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    values = torch.randn(global_shape, dtype=torch.bfloat16)
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    expected = torch.zeros(shared_shape, dtype=torch.bfloat16)
    expected[:, :72] = values
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_oob_zero_fill_int32_data_consistency():
    kernel = tilelang.compile(
        _make_oob_tme_load((20,), (32,), dtype="int32"),
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    values = torch.arange(20, dtype=torch.int32)
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    expected = torch.zeros((32,), dtype=torch.int32)
    expected[:20] = values
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
