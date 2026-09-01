import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.transform import PassConfigKey


def _explicit_ldg_stg_kernel(bits):
    lanes = bits // 32
    ldg = getattr(T, f"ldg{bits}")
    stg = getattr(T, f"stg{bits}")

    @T.prim_func
    def kernel(
        src: T.Tensor((8,), T.float32),
        dst: T.Tensor((8,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            for i in T.serial(8):
                dst[i] = 0.0
            value = ldg(src[0:lanes])
            stg(dst[0:lanes], value)

    return kernel


def _vector_copy_kernel(lanes, dtype=T.float32):
    @T.prim_func
    def kernel(
        src: T.Tensor((256,), dtype),
        dst: T.Tensor((256,), dtype),
    ):
        with T.Kernel(1, threads=32):
            tid = T.get_thread_binding()
            for j in T.vectorized(lanes):
                dst[tid * lanes + j] = src[tid * lanes + j]

    return kernel


@T.prim_func
def _explicit_predicated_ldg_stg_kernel(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=32):
        tid = T.get_thread_binding()
        pred = tid < 16
        for j in T.vectorized(4):
            dst[tid * 4 + j] = 0.0
        value = T.ldg128(src[tid * 4 : tid * 4 + 4], pred=pred)
        T.stg128(dst[tid * 4 : tid * 4 + 4], value, pred=pred)


@T.prim_func
def _predicated_vector_copy_kernel(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=32):
        tid = T.get_thread_binding()
        for j in T.vectorized(4):
            index = tid * 4 + j
            dst[index] = T.if_then_else(tid < 16, src[index], T.float32(0))


@T.prim_func
def _vector_copy_256_ir(
    src: T.Tensor((256,), T.float32),
    dst: T.Tensor((256,), T.float32),
):
    for i in T.thread_binding(32, "threadIdx.x"):
        for j in T.vectorized(8):
            dst[i * 8 + j] = src[i * 8 + j]


@T.prim_func
def _async_copy_with_ldg_stg_kernel(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=32):
        smem = T.alloc_shared((128,), T.float32)
        T.async_copy(src, smem)
        T.ptx_wait_group(0)
        T.sync_threads()
        T.copy(smem, dst)


def _lower_with_configs(kernel, configs):
    target = tvm.target.Target({"kind": "musa"})
    with target, tvm.transform.PassContext(config=configs):
        return tilelang.lower(
            kernel,
            target=target,
            enable_device_compile=False,
        )


def _has_intrinsic(mod, name):
    found = [False]

    def visitor(obj):
        if isinstance(obj, tvm.tirx.Call) and getattr(obj.op, "name", "") == name:
            found[0] = True

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return found[0]


@pytest.mark.parametrize("bits", [32, 64, 128, 256])
@tilelang.testing.requires_musa
def test_musa_explicit_ldg_stg_codegen(bits):
    artifact = _lower_with_configs(_explicit_ldg_stg_kernel(bits), {})
    source = artifact.kernel_source

    assert "#include <tl_templates/musa/common/copy.h>" in source
    assert f"tl::load_global_{bits}(" in source
    assert f"tl::store_global_{bits}(" in source

    kernel = tilelang.compile(
        _explicit_ldg_stg_kernel(bits),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(8, dtype=torch.float32)
    out = kernel(src.to("musa"))
    torch.musa.synchronize()
    expected = torch.zeros(8, dtype=torch.float32)
    expected[: bits // 32] = src[: bits // 32]
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_explicit_predicated_ldg_stg_codegen():
    artifact = _lower_with_configs(_explicit_predicated_ldg_stg_kernel, {})
    source = artifact.kernel_source

    assert "tl::load_global_128_conditional(" in source
    assert "tl::store_global_128_conditional(" in source

    kernel = tilelang.compile(
        _explicit_predicated_ldg_stg_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(128, dtype=torch.float32)
    out = kernel(src.to("musa"))
    torch.musa.synchronize()
    expected = torch.zeros(128, dtype=torch.float32)
    expected[:64] = src[:64]
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "lanes,dtype",
    [(1, T.float32), (2, T.float32), (4, T.float32), (4, T.float8_e4m3fn)],
)
@tilelang.testing.requires_musa
def test_musa_automatic_lower_ldg_stg_codegen(lanes, dtype):
    bits = lanes * (8 if dtype is T.float8_e4m3fn else 32)
    artifact = _lower_with_configs(
        _vector_copy_kernel(lanes, dtype),
        {PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True},
    )
    source = artifact.kernel_source

    assert f"tl::load_global_{bits}(" in source
    assert f"tl::store_global_{bits}(" in source


@tilelang.testing.requires_musa
def test_musa_lower_ldg_stg_pass_supports_256_bits():
    target = tvm.target.Target({"kind": "musa"})
    mod = tvm.IRModule.from_expr(
        _vector_copy_256_ir.with_attr("global_symbol", "main")
    )
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    mod = tilelang.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(
        config={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True}
    ):
        mod = tilelang.musa.transform.LowerLDGSTG()(mod)

    assert _has_intrinsic(mod, "tl.ldg256")
    assert _has_intrinsic(mod, "tl.stg256")


@tilelang.testing.requires_musa
def test_musa_lower_ldg_stg_default_off():
    artifact = _lower_with_configs(_vector_copy_kernel(4), {})
    source = artifact.kernel_source

    assert "tl::load_global_128(" not in source
    assert "tl::store_global_128(" not in source


@tilelang.testing.requires_musa
def test_musa_predicated_lower_ldg_stg_codegen():
    artifact = _lower_with_configs(
        _predicated_vector_copy_kernel,
        {PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True},
    )
    source = artifact.kernel_source

    assert "tl::load_global_128_conditional(" in source

    kernel = tilelang.compile(
        _predicated_vector_copy_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True},
    )
    src = torch.arange(128, dtype=torch.float32)
    out = kernel(src.to("musa"))
    torch.musa.synchronize()
    expected = torch.zeros(128, dtype=torch.float32)
    expected[:64] = src[:64]
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_async_copy_is_not_rewritten_to_ldg():
    artifact = _lower_with_configs(
        _async_copy_with_ldg_stg_kernel,
        {PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True},
    )
    source = artifact.kernel_source

    assert "tl::cp_async_gs<16>" in source
    assert "tl::load_global_" not in source


@tilelang.testing.requires_musa
@pytest.mark.parametrize("dtype", [T.float32, T.float8_e4m3fn])
def test_musa_automatic_ldg_stg_runtime_values(dtype):
    kernel = tilelang.compile(
        _vector_copy_kernel(8, dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
        pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True},
    )

    src_cpu = torch.arange(256, dtype=torch.float32)
    if dtype is T.float8_e4m3fn:
        src_cpu = (src_cpu % 8).to(torch.float8_e4m3fn)
    dst = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(dst.cpu(), src_cpu, atol=0, rtol=0)
