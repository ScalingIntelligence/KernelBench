import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import torch.nn.functional as F
from tilelang import tvm


def _make_tme_im2col(
    *,
    input_size=4,
    channels=8,
    rows=4,
    nhw_step=0,
    c_step=0,
    kernel_size=3,
    stride=1,
    dilation=1,
    padding=0,
    dtype="float32",
):
    @T.prim_func
    def kernel(
        A: T.Tensor((1, input_size, input_size, channels), dtype),
        C: T.Tensor((rows, channels), dtype),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((rows, channels), dtype)
            T.im2col(
                A,
                smem,
                nhw_step,
                c_step,
                kernel_size,
                stride,
                dilation,
                padding,
            )
            for i, j in T.Parallel(rows, channels):
                C[i, j] = smem[i, j]

    return kernel


def _make_all_taps_kernel(
    *,
    input_size,
    channels,
    kernel_size,
    stride,
    dilation,
    padding,
    dtype,
):
    output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    rows = output_size * output_size
    taps = kernel_size * kernel_size

    @T.prim_func
    def kernel(
        A: T.Tensor((1, input_size, input_size, channels), dtype),
        C: T.Tensor((taps, rows, channels), dtype),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((rows, channels), dtype)
            # c_step is a tile index. Keeping it as a loop expression checks
            # dynamic weight-position encoding and barrier phase alternation.
            for tap in T.serial(taps):
                T.im2col(
                    A,
                    smem,
                    0,
                    tap,
                    kernel_size,
                    stride,
                    dilation,
                    padding,
                )
                for i, j in T.Parallel(rows, channels):
                    C[tap, i, j] = smem[i, j]

    return kernel


def _lower(program, *, arch="mp_31", compile_device=False):
    target = tvm.target.Target({"kind": "musa", "arch": arch})
    with target:
        return tilelang.lower(program, target=target, enable_device_compile=compile_device)


def _reference(values, kernel_size, stride, dilation, padding):
    n, h, w, channels = values.shape
    assert n == 1 and h == w
    columns = F.unfold(
        values.permute(0, 3, 1, 2),
        kernel_size=kernel_size,
        dilation=dilation,
        padding=padding,
        stride=stride,
    )
    rows = columns.shape[-1]
    return (
        columns.reshape(n, channels, kernel_size, kernel_size, rows)
        .permute(2, 3, 4, 1, 0)
        .reshape(kernel_size * kernel_size, rows, channels)
    )


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_im2col_codegen_and_mcc():
    source = _lower(_make_tme_im2col(), compile_device=True).kernel_source
    # A complete (rows, channels) tile is one MP31 im2col transaction.
    assert source.count("tl::tme_load_im2col(") == 1
    assert "tl::tme_barrier_add_trans" in source
    assert "tl::tme_barrier_arrive" in source
    assert "tl::tme_barrier_wait" in source


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_im2col_rejects_unrepresentable_channel_tile():
    # A 12-element float16 channel tile is 24 bytes and cannot be represented
    # by the MP31 TME im2col instruction.
    with pytest.raises(tvm.error.InternalError, match="16-byte aligned"):
        _lower(_make_tme_im2col(channels=12, rows=4, dtype="float16"))


@pytest.mark.parametrize(
    "input_size,kernel_size,stride,dilation,padding",
    [
        (4, 3, 1, 1, 0),
        (4, 3, 1, 1, 1),
        (7, 3, 2, 2, 1),
    ],
)
@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_im2col_data_consistency(input_size, kernel_size, stride, dilation, padding):
    channels = 8
    values = torch.arange(input_size * input_size * channels, dtype=torch.float32).reshape(1, input_size, input_size, channels)
    program = _make_all_taps_kernel(
        input_size=input_size,
        channels=channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        dtype="float32",
    )
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    expected = _reference(values, kernel_size, stride, dilation, padding)
    torch.testing.assert_close(result.cpu(), expected)


@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_im2col_nonzero_tile_steps():
    values = torch.arange(4 * 4 * 8, dtype=torch.float32).reshape(1, 4, 4, 8)
    program = _make_tme_im2col(
        input_size=4,
        channels=8,
        rows=4,
        nhw_step=2,
        c_step=4,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=1,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    expected = _reference(values, 3, 1, 1, 1)[4, 8:12]
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize(
    "tilelang_dtype,torch_dtype",
    [("bfloat16", torch.bfloat16), ("int32", torch.int32)],
)
@tilelang.testing.requires_musa_compute_version_le(3, 1)
def test_mp31_tme_im2col_descriptor_dtypes(tilelang_dtype, torch_dtype):
    values = torch.arange(4 * 4 * 8, dtype=torch.float32).reshape(1, 4, 4, 8).to(torch_dtype)
    program = _make_all_taps_kernel(
        input_size=4,
        channels=8,
        kernel_size=3,
        stride=1,
        dilation=1,
        padding=1,
        dtype=tilelang_dtype,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    result = kernel(values.to("musa"))
    torch.musa.synchronize()
    expected = _reference(values.float(), 3, 1, 1, 1).to(torch_dtype)
    torch.testing.assert_close(result.cpu(), expected)


if __name__ == "__main__":
    tilelang.testing.main()
