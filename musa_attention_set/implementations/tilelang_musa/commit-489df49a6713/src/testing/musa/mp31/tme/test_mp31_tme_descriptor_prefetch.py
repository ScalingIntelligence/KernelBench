import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


PREFETCH_CONFIG = {"tl.enable_musa_tma_prefetch": True}


@T.prim_func
def _tme_load_store(
    A: T.Tensor((8, 32), "float32"), C: T.Tensor((8, 32), "float32")
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((8, 32), "float32")
        barrier = T.alloc_barrier(128)
        T.tma_copy(A[0, 0], smem, barrier=barrier)
        T.barrier_arrive(barrier)
        T.barrier_wait(barrier, 0)
        T.tma_copy(smem, C[0, 0])
        T.tma_store_wait(0)


@T.prim_func
def _tme_prefetch_dedup(A: T.Tensor((8, 32), "float32")):
    with T.Kernel(1, threads=128):
        smem0 = T.alloc_shared((8, 32), "float32")
        smem1 = T.alloc_shared((8, 32), "float32")
        barrier0 = T.alloc_barrier(128)
        barrier1 = T.alloc_barrier(128)
        T.tma_copy(A[0, 0], smem0, barrier=barrier0)
        T.tma_copy(A[0, 0], smem1, barrier=barrier1)
        T.barrier_arrive(barrier0)
        T.barrier_arrive(barrier1)
        T.barrier_wait(barrier0, 0)
        T.barrier_wait(barrier1, 0)


def _lower(program, enable_prefetch):
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    config = PREFETCH_CONFIG if enable_prefetch else {}
    with target, tvm.transform.PassContext(config=config):
        return tilelang.lower(
            program, target=target, enable_device_compile=False
        )


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_descriptor_prefetch_codegen():
    enabled_source = _lower(_tme_load_store, True).kernel_source
    disabled_source = _lower(_tme_load_store, False).kernel_source
    assert enabled_source.count("tl::prefetch_tma_descriptor(") == 2
    assert "tl::prefetch_tma_descriptor(" not in disabled_source


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_descriptor_prefetch_dedup():
    source = _lower(_tme_prefetch_dedup, True).kernel_source
    assert source.count("tl::prefetch_tma_descriptor(") == 1


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_descriptor_prefetch_data_consistency():
    kernel = tilelang.compile(
        _tme_load_store,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
        pass_configs=PREFETCH_CONFIG,
    )
    values = torch.randn((8, 32), dtype=torch.float32)
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(result.cpu(), values)
