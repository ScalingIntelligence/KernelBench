import tilelang.testing
import tilelang.language as T
import tilelang.musa.language as musa_language
from tilelang.backend.target import determine_target
from tilelang.musa.target import musa_arch_to_compute_version, normalize_musa_arch, normalize_musa_target
from tvm.target import Target


def test_musa_language_frontend():
    assert T.__tilelang_dialect__ == "musa"
    assert musa_language.__tilelang_dialect__ == "musa"
    assert T.device_assert is musa_language.device_assert


@tilelang.testing.requires_musa
def test_normalize_musa_arch():
    assert normalize_musa_arch(None) is None
    assert normalize_musa_arch("invalid") is None
    assert musa_arch_to_compute_version("mp_31") == (3, 1)
    assert musa_arch_to_compute_version("invalid") is None


@tilelang.testing.requires_musa
def test_normalize_musa_target():
    target = normalize_musa_target({"kind": "musa"})
    assert target is not None
    assert target.kind.name == "musa"

    parsed = normalize_musa_target(Target("musa"))
    assert parsed is not None
    assert parsed.kind.name == "musa"

    assert normalize_musa_target({"kind": "llvm"}) is None


@tilelang.testing.requires_musa
def test_auto_detect_musa_target():
    target = determine_target("auto", return_object=True)
    assert target.kind.name == "musa"
    assert musa_arch_to_compute_version(target.attrs.get("arch")) != (0, 0)
