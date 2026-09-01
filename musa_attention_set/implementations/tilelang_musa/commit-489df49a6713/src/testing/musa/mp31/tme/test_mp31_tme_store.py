import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


def _make_tme_store(rank_shape):
    shape = tuple(rank_shape)
    starts = (0,) * len(shape)

    @T.prim_func
    def kernel(A: T.Tensor(shape, "float32"), C: T.Tensor(shape, "float32")):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared(shape, "float32")
            if len(shape) == 1:
                for i0 in T.Parallel(shape[0]):
                    smem[i0] = A[i0]
            elif len(shape) == 2:
                for i0, i1 in T.Parallel(shape[0], shape[1]):
                    smem[i0, i1] = A[i0, i1]
            elif len(shape) == 3:
                for i0, i1, i2 in T.Parallel(shape[0], shape[1], shape[2]):
                    smem[i0, i1, i2] = A[i0, i1, i2]
            elif len(shape) == 4:
                for i0, i1, i2, i3 in T.Parallel(
                    shape[0], shape[1], shape[2], shape[3]
                ):
                    smem[i0, i1, i2, i3] = A[i0, i1, i2, i3]
            else:
                for i0, i1, i2, i3, i4 in T.Parallel(
                    shape[0], shape[1], shape[2], shape[3], shape[4]
                ):
                    smem[i0, i1, i2, i3, i4] = A[i0, i1, i2, i3, i4]
            T.sync_threads()
            T.tma_copy(smem, C[starts])
            T.tma_store_wait(0)

    return kernel


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_store_ranks_codegen():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    shapes = [(128,), (8, 32), (4, 8, 32), (2, 4, 8, 32), (2, 2, 4, 8, 32)]
    for shape in shapes:
        with target:
            artifact = tilelang.lower(
                _make_tme_store(shape), target=target, enable_device_compile=False
            )
        source = artifact.kernel_source
        assert "tl::tme_store(" in source
        assert "tl::tme_store_commit()" in source
        assert "tl::tme_store_read_wait()" in source
        assert "tl_templates/musa/mp31/tme.h" in source


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_store_data_consistency():
    shapes = [(128,), (8, 32), (4, 8, 32), (2, 4, 8, 32), (2, 2, 4, 8, 32)]
    for shape in shapes:
        kernel = tilelang.compile(
            _make_tme_store(shape),
            out_idx=[1],
            target={"kind": "musa", "arch": "mp_31"},
            execution_backend="tvm_ffi",
        )
        values = torch.randn(shape, dtype=torch.float32)
        result = kernel(values.to("musa"))
        torch.musa.synchronize()
        torch.testing.assert_close(result.cpu(), values)
