from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


_build_musa = global_func_device_codegen("target.build.musa")


def _is_musa_target(target: Target) -> bool:
    return target.kind.name == "musa"


register_device_codegen(
    "musa",
    DeviceCodegen(
        "musa",
        build=_build_musa,
        build_without_compile=_build_musa,
        supports_target=_is_musa_target,
    ),
    override=True,
)
