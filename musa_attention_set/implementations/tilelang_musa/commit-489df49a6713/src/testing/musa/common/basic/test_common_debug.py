import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _debug_print_kernel():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            tx = T.get_thread_binding()
            T.print(tx, msg="tx")
            T.print(msg="hello")

    return kernel


def _device_assert_kernel():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            tx = T.get_thread_binding()
            T.device_assert(tx == tx, no_stack_info=True)
            T.device_assert(tx == tx, "musa debug assert")

    return kernel


def _debug_typed_print_kernel():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            tx = T.get_thread_binding()
            T.print(T.cast(tx == 0, T.bool), msg="bool")
            T.print(T.cast(tx, T.int64), msg="int64")
            T.print(T.cast(tx, T.uint16), msg="uint16")
            T.print(T.cast(tx, T.float64), msg="float64")

    return kernel


@tilelang.testing.requires_musa
def test_musa_common_debug_print_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _debug_print_kernel(),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/debug.h>" in artifact.kernel_source
    assert "debug_print_var" in artifact.kernel_source
    assert "debug_print_msg" in artifact.kernel_source


@tilelang.testing.requires_musa
def test_musa_common_device_assert_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _device_assert_kernel(),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/debug.h>" in artifact.kernel_source
    assert "device_assert(" in artifact.kernel_source
    assert "device_assert_with_msg(" in artifact.kernel_source


@tilelang.testing.requires_musa
def test_musa_common_device_assert_runtime_no_trigger():
    kernel = tilelang.compile(
        _device_assert_kernel(),
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    kernel()


@tilelang.testing.requires_musa
def test_musa_common_debug_typed_print_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _debug_typed_print_kernel(),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "#include <tl_templates/musa/common/debug.h>" in artifact.kernel_source
    assert "debug_print_var" in artifact.kernel_source
