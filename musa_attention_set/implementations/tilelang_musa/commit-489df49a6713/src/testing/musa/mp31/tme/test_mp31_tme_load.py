import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


@T.prim_func
def _tme_load_1d(A: T.Tensor((128,), "float32"), C: T.Tensor((128,), "float32")):
    with T.Kernel(1, threads=32):
        smem = T.alloc_shared((128,), "float32")
        barrier = T.alloc_barrier(32)
        T.tma_copy(A[0], smem, barrier=barrier)
        T.barrier_arrive(barrier)
        T.barrier_wait(barrier, 0)
        for i in T.Parallel(128):
            C[i] = smem[i]


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_load_codegen():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        artifact = tilelang.lower(
            _tme_load_1d, target=target, enable_device_compile=False
        )
    source = artifact.kernel_source
    assert "tl_templates/musa/mp31/tme.h" in source
    assert "tl::tme_load(" in source
    assert "__musa_tme_ld_tile_1d" not in source
    assert "tl::tme_barrier_add_trans" in source


def _make_tme_load(rank_shape):
    shape = tuple(rank_shape)
    starts = (0,) * len(shape)

    @T.prim_func
    def kernel(A: T.Tensor(shape, "float32")):
        with T.Kernel(1, threads=32):
            smem = T.alloc_shared(shape, "float32")
            barrier = T.alloc_barrier(32)
            T.tma_copy(A[starts], smem, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)

    return kernel


def _make_tme_load_copy(rank_shape):
    shape = tuple(rank_shape)
    starts = (0,) * len(shape)

    @T.prim_func
    def kernel(A: T.Tensor(shape, "float32"), C: T.Tensor(shape, "float32")):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared(shape, "float32")
            barrier = T.alloc_barrier(128)
            T.tma_copy(A[starts], smem, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            if len(shape) == 1:
                for i0 in T.Parallel(shape[0]):
                    C[i0] = smem[i0]
            elif len(shape) == 2:
                for i0, i1 in T.Parallel(shape[0], shape[1]):
                    C[i0, i1] = smem[i0, i1]
            elif len(shape) == 3:
                for i0, i1, i2 in T.Parallel(shape[0], shape[1], shape[2]):
                    C[i0, i1, i2] = smem[i0, i1, i2]
            elif len(shape) == 4:
                for i0, i1, i2, i3 in T.Parallel(
                    shape[0], shape[1], shape[2], shape[3]
                ):
                    C[i0, i1, i2, i3] = smem[i0, i1, i2, i3]
            else:
                for i0, i1, i2, i3, i4 in T.Parallel(
                    shape[0], shape[1], shape[2], shape[3], shape[4]
                ):
                    C[i0, i1, i2, i3, i4] = smem[i0, i1, i2, i3, i4]

    return kernel


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_load_ranks_codegen():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    shapes = [(128,), (8, 32), (4, 8, 32), (2, 4, 8, 32), (2, 2, 4, 8, 32)]
    for shape in shapes:
        with target:
            artifact = tilelang.lower(
                _make_tme_load(shape), target=target, enable_device_compile=False
            )
        source = artifact.kernel_source
        assert "tl::tme_load(" in source
        assert "tl_templates/musa/mp31/tme.h" in source

@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_load_data_consistency():
    shapes = [(128,), (8, 32), (4, 8, 32), (2, 4, 8, 32), (2, 2, 4, 8, 32)]
    for shape in shapes:
        kernel = tilelang.compile(
            _make_tme_load_copy(shape),
            out_idx=[1],
            target={"kind": "musa", "arch": "mp_31"},
            execution_backend="tvm_ffi",
        )
        values = torch.randn(shape, dtype=torch.float32)
        result = kernel(values.to("musa"))
        torch.musa.synchronize()
        torch.testing.assert_close(result.cpu(), values)
