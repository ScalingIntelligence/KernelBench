import tilelang.testing
import torch

from tilelang import tvm
import tilelang
import tilelang.language as T

_NUM_THREADS = 1024
_DEFAULT_WARPS_PER_GROUP = 4
_SHUFFLE_ELECT_GROUP_THREADS = 256


def _make_empty_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=1):
            T.evaluate(0)

    return main


@T.prim_func
def _intrin_kernel(A: T.Tensor((_NUM_THREADS, 6), "int32")):
    with T.Kernel(1, threads=_NUM_THREADS):
        tx = T.get_thread_binding()
        A[tx, 0] = T.get_lane_idx()
        A[tx, 1] = T.get_warp_idx_sync()
        A[tx, 2] = T.get_warp_idx()
        A[tx, 3] = T.get_warp_group_idx()
        A[tx, 4] = T.if_then_else(T.shuffle_elect(0), 1, 0)
        A[tx, 5] = T.if_then_else(T.shuffle_elect(_SHUFFLE_ELECT_GROUP_THREADS), 1, 0)


@tilelang.testing.requires_musa
def test_musa_common_intrin_template_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(_make_empty_kernel(), target=target, enable_device_compile=False)

    assert artifact.kernel_source is not None
    assert "#include <musa.h>" in artifact.kernel_source
    assert "#include <tl_templates/musa/common/intrin.h>" in artifact.kernel_source
    assert "__global__" in artifact.kernel_source


@tilelang.testing.requires_musa
def test_musa_common_intrin_runtime_values():
    kernel = tilelang.compile(
        _intrin_kernel,
        out_idx=[0],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    out = kernel()
    torch.musa.synchronize()

    warp_idx = out[:, 1].cpu()
    warp_size = int((warp_idx == 0).sum().item())

    idx = torch.arange(_NUM_THREADS, dtype=torch.int32)
    ref = torch.stack(
        [
            idx % warp_size,
            idx // warp_size,
            idx // warp_size,
            idx // (warp_size * _DEFAULT_WARPS_PER_GROUP),
        ],
        dim=1,
    )
    out_cpu = out.cpu()
    torch.testing.assert_close(out_cpu[:, :4], ref)

    block_elect = out_cpu[:, 4]
    assert int(block_elect.sum().item()) == 1
    assert torch.all((block_elect == 0) | (block_elect == 1))

    group_elect = out_cpu[:, 5].reshape(-1, _SHUFFLE_ELECT_GROUP_THREADS)
    torch.testing.assert_close(
        group_elect.sum(dim=1),
        torch.ones(_NUM_THREADS // _SHUFFLE_ELECT_GROUP_THREADS, dtype=torch.int64),
    )
    assert torch.all((group_elect == 0) | (group_elect == 1))
