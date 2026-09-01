import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_NUM_THREADS = 128
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


@T.prim_func
def _warp_reduce_kernel(
    A: T.Tensor((_NUM_THREADS,), "float32"),
    B: T.Tensor((_NUM_THREADS,), "int32"),
    OutF: T.Tensor((_NUM_THREADS, 3), "float32"),
    OutI: T.Tensor((_NUM_THREADS, 3), "int32"),
):
    with T.Kernel(1, threads=_NUM_THREADS):
        tx = T.get_thread_binding()
        OutF[tx, 0] = T.warp_reduce_sum(A[tx])
        OutF[tx, 1] = T.warp_reduce_max(A[tx])
        OutF[tx, 2] = T.warp_reduce_min(A[tx])
        OutI[tx, 0] = T.get_warp_idx_sync()
        OutI[tx, 1] = T.warp_reduce_bitand(B[tx])
        OutI[tx, 2] = T.warp_reduce_bitor(B[tx])


def _make_reduce_sum_kernel():
    @T.prim_func
    def kernel(
        X: T.Tensor((1, 1024), "float32"),
        Out: T.Tensor((1,), "float32"),
    ):
        with T.Kernel(1, threads=_NUM_THREADS):
            x_frag = T.alloc_fragment((1, 1024), "float32")
            out_frag = T.alloc_fragment((1,), "float32")
            T.annotate_layout(
                {
                    x_frag: T.Fragment(
                        x_frag.shape,
                        forward_fn=lambda i, j: (j // 8, j % 8),
                    ),
                    out_frag: T.Fragment(
                        out_frag.shape,
                        forward_fn=lambda i, rep: (rep, 0),
                        replicate=_NUM_THREADS,
                    ),
                }
            )
            for i, j in T.Parallel(1, 1024):
                x_frag[i, j] = X[i, j]
            T.reduce_sum(x_frag, out_frag, dim=1)
            for i in T.Parallel(1):
                Out[i] = out_frag[i]

    return kernel


def _make_reduce_sum_kernel_for_dtype(dtype):
    @T.prim_func
    def kernel(
        X: T.Tensor((1, 1024), dtype),
        Out: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1, threads=_NUM_THREADS):
            x_frag = T.alloc_fragment((1, 1024), dtype)
            out_frag = T.alloc_fragment((1,), dtype)
            T.annotate_layout(
                {
                    x_frag: T.Fragment(
                        x_frag.shape,
                        forward_fn=lambda i, j: (j // 8, j % 8),
                    ),
                    out_frag: T.Fragment(
                        out_frag.shape,
                        forward_fn=lambda i, rep: (rep, 0),
                        replicate=_NUM_THREADS,
                    ),
                }
            )
            for i, j in T.Parallel(1, 1024):
                x_frag[i, j] = X[i, j]
            T.reduce_sum(x_frag, out_frag, dim=1)
            for i in T.Parallel(1):
                Out[i] = out_frag[i]

    return kernel


def _torch_dtype(tilelang_dtype):
    return {
        T.float32: torch.float32,
        T.int64: torch.int64,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
    }[tilelang_dtype]


def _half_warp_reduce_kernel(dtype):
    @T.prim_func
    def kernel(
        A: T.Tensor((_NUM_THREADS,), dtype),
        Out: T.Tensor((_NUM_THREADS, 2), dtype),
        Warp: T.Tensor((_NUM_THREADS,), "int32"),
    ):
        with T.Kernel(1, threads=_NUM_THREADS):
            tx = T.get_thread_binding()
            Out[tx, 0] = T.warp_reduce_max(A[tx])
            Out[tx, 1] = T.warp_reduce_min(A[tx])
            Warp[tx] = T.get_warp_idx_sync()

    return kernel


@tilelang.testing.requires_musa
def test_musa_reduce_template_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        warp_artifact = tilelang.lower(
            _warp_reduce_kernel,
            target=target,
            enable_device_compile=False,
        )
        with tvm.transform.PassContext(config=_PASS_CONFIGS):
            reduce_artifact = tilelang.lower(
                _make_reduce_sum_kernel(),
                target=target,
                enable_device_compile=False,
            )

    assert warp_artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/reduce.h>" in warp_artifact.kernel_source
    assert "tl::warp_reduce_sum(" in warp_artifact.kernel_source
    assert "tl::warp_reduce_max(" in warp_artifact.kernel_source
    assert "tl::warp_reduce_min(" in warp_artifact.kernel_source
    assert "tl::warp_reduce_bitand(" in warp_artifact.kernel_source
    assert "tl::warp_reduce_bitor(" in warp_artifact.kernel_source

    assert reduce_artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/reduce.h>" in reduce_artifact.kernel_source
    assert "tl::AllReduce<tl::SumOp" in reduce_artifact.kernel_source


@tilelang.testing.requires_musa
def test_musa_warp_reduce_runtime_values():
    kernel = tilelang.compile(
        _warp_reduce_kernel,
        out_idx=[2, 3],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    a_cpu = torch.arange(_NUM_THREADS, dtype=torch.float32)
    b_cpu = torch.where(
        torch.arange(_NUM_THREADS, dtype=torch.int32) % 3 == 0,
        torch.tensor(3, dtype=torch.int32),
        torch.tensor(1, dtype=torch.int32),
    )

    out_f, out_i = kernel(a_cpu.to("musa"), b_cpu.to("musa"))
    torch.musa.synchronize()

    out_f = out_f.cpu()
    out_i = out_i.cpu()

    warp_idx = out_i[:, 0]
    ref_f = torch.empty_like(out_f)
    ref_i = torch.empty_like(out_i[:, 1:])
    for warp in torch.unique(warp_idx):
        mask = warp_idx == warp
        a_group = a_cpu[mask]
        b_group = b_cpu[mask]
        ref_f[mask, 0] = a_group.sum()
        ref_f[mask, 1] = a_group.max()
        ref_f[mask, 2] = a_group.min()

        bitand = b_group[0].clone()
        bitor = b_group[0].clone()
        for value in b_group[1:]:
            bitand = bitand & value
            bitor = bitor | value
        ref_i[mask, 0] = bitand
        ref_i[mask, 1] = bitor

    torch.testing.assert_close(out_f, ref_f, atol=0, rtol=0)
    torch.testing.assert_close(out_i[:, 1:], ref_i, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.int64])
@tilelang.testing.requires_musa
def test_musa_half_warp_reduce_max_min_runtime_values(dtype):
    kernel = tilelang.compile(
        _half_warp_reduce_kernel(dtype),
        out_idx=[1, 2],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    torch_dtype = _torch_dtype(dtype)
    values = (torch.arange(_NUM_THREADS, dtype=torch.float32) % 17 - 8).to(torch_dtype)

    out, warp = kernel(values.to("musa"))
    torch.musa.synchronize()

    out = out.cpu()
    warp = warp.cpu()
    ref = torch.empty_like(out)
    for warp_id in torch.unique(warp):
        mask = warp == warp_id
        group = values[mask]
        ref[mask, 0] = group.max()
        ref[mask, 1] = group.min()

    torch.testing.assert_close(out, ref, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [T.float32, T.int64])
@tilelang.testing.requires_musa
def test_musa_reduce_sum_runtime_values(dtype):
    kernel = tilelang.compile(
        _make_reduce_sum_kernel_for_dtype(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
        pass_configs=_PASS_CONFIGS,
    )

    x_cpu = torch.arange(1024, dtype=_torch_dtype(dtype)).reshape(1, 1024)
    out = kernel(x_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(out.cpu(), x_cpu.sum(dim=1), atol=0, rtol=0)
