from __future__ import annotations

import dataclasses
import os
import shutil
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import tvm_ffi
import tvm_ffi.cpp
from tvm_ffi.utils import FileLock

from . import env as jit_env
from .cpp_ext import (
    generate_compile_commands_for_op,
    generate_ninja_build_for_op,
    run_ninja,
)


def write_if_different(path: Path, content: str) -> bool:
    if path.exists() and path.read_text() == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def get_tmpdir() -> Path:
    tmpdir = jit_env.MATE_JIT_DIR / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    return tmpdir


def default_aot_path(name: str, base_dir: Optional[Path] = None) -> Path:
    root = base_dir or jit_env.MATE_AOT_DIR
    return root / name / f"{name}.so"


class MissingJITCacheError(RuntimeError):
    """Raised when runtime JIT is disabled and no matching AOT module exists."""

    def __init__(self, message: str, spec: Optional["JitSpec"] = None):
        self.spec = spec
        super().__init__(message)


@dataclasses.dataclass
class JitSpecStatus:
    name: str
    created_at: datetime
    is_compiled: bool
    library_path: Optional[Path]
    sources: List[Path]
    is_aot: bool
    is_jit_compiled: bool
    aot_path: Path
    jit_library_path: Path
    build_dir: Path
    ninja_path: Path
    num_generated_sources: int

    @property
    def status(self) -> str:
        if self.is_aot:
            return "AOT"
        if self.is_jit_compiled:
            return "JIT Compiled"
        return "Not Compiled"

    @property
    def kind(self) -> str:
        return self.status

    @property
    def num_sources(self) -> int:
        return len(self.sources)


class JitSpecRegistry:
    """Global registry that tracks JIT specs created in the current process."""

    def __init__(self) -> None:
        self._specs: Dict[str, JitSpec] = {}
        self._creation_times: Dict[str, datetime] = {}

    def register(self, spec: "JitSpec") -> None:
        existing = self._specs.get(spec.name)
        if existing is None:
            self._specs[spec.name] = spec
            self._creation_times[spec.name] = datetime.now()
            return
        # The same logical dispatch can be re-created from different build roots
        # (for example AOT packaging vs. runtime JIT cache). Keep the first
        # registration so status tracking remains stable across those contexts.
        if existing != spec:
            return

    def get_all_specs(self) -> Dict[str, "JitSpec"]:
        return self._specs.copy()

    def get_spec_status(self, name: str) -> Optional[JitSpecStatus]:
        spec = self._specs.get(name)
        if spec is None:
            return None
        library_path = spec.get_library_path() if spec.is_compiled else None
        return JitSpecStatus(
            name=name,
            created_at=self._creation_times[name],
            is_compiled=spec.is_compiled,
            library_path=library_path,
            sources=spec.sources,
            is_aot=spec.is_aot,
            is_jit_compiled=spec.is_jit_compiled and not spec.is_aot,
            aot_path=spec.resolved_aot_path,
            jit_library_path=spec.jit_library_path,
            build_dir=spec.build_dir,
            ninja_path=spec.ninja_path,
            num_generated_sources=len(spec.generated_sources),
        )

    def get_all_statuses(self) -> List[JitSpecStatus]:
        statuses: List[JitSpecStatus] = []
        for name in self._specs:
            status = self.get_spec_status(name)
            if status is not None:
                statuses.append(status)
        return statuses

    def get_stats(self) -> Dict[str, int]:
        statuses = self.get_all_statuses()
        return {
            "total": len(statuses),
            "compiled": sum(1 for status in statuses if status.is_compiled),
            "aot": sum(1 for status in statuses if status.is_aot),
            "jit_compiled": sum(1 for status in statuses if status.is_jit_compiled),
            "not_compiled": sum(1 for status in statuses if not status.is_compiled),
        }


jit_spec_registry = JitSpecRegistry()


@dataclasses.dataclass
class JitSpec:
    name: str
    sources: List[Path]
    extra_cflags: Optional[List[str]] = None
    extra_cuda_cflags: Optional[List[str]] = None
    extra_ldflags: Optional[List[str]] = None
    extra_include_dirs: Optional[List[Path]] = None
    generated_sources: Dict[Path, str] = dataclasses.field(default_factory=dict)
    aot_path: Optional[Path] = None

    @property
    def build_dir(self) -> Path:
        return jit_env.MATE_JIT_DIR / self.name

    @property
    def ninja_path(self) -> Path:
        return self.build_dir / "build.ninja"

    @property
    def jit_library_path(self) -> Path:
        return self.build_dir / f"{self.name}.so"

    @property
    def lock_path(self) -> Path:
        return get_tmpdir() / f"{self.name}.lock"

    @property
    def is_aot(self) -> bool:
        return self.aot_path is not None and self.aot_path.exists()

    @property
    def is_jit_compiled(self) -> bool:
        return self.jit_library_path.exists()

    @property
    def is_compiled(self) -> bool:
        return self.is_aot or self.is_jit_compiled

    @property
    def resolved_aot_path(self) -> Path:
        return self.aot_path or default_aot_path(self.name)

    def get_library_path(self) -> Path:
        if self.is_aot:
            return self.aot_path  # type: ignore[return-value]
        return self.jit_library_path

    def materialize_sources(self) -> None:
        for path, content in self.generated_sources.items():
            write_if_different(path, content)

    def write_ninja(self) -> None:
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.materialize_sources()
        content = generate_ninja_build_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_ldflags=self.extra_ldflags,
            extra_include_dirs=self.extra_include_dirs,
        )
        write_if_different(self.ninja_path, content)

    def load(self, library_path: Path):
        return tvm_ffi.load_module(str(library_path))

    def get_compile_commands(self) -> List[dict]:
        return generate_compile_commands_for_op(
            name=self.name,
            sources=self.sources,
            extra_cflags=self.extra_cflags,
            extra_cuda_cflags=self.extra_cuda_cflags,
            extra_include_dirs=self.extra_include_dirs,
        )

    def build(self, verbose: bool = False, need_lock: bool = True) -> Path:
        if jit_env.disable_jit_enabled():
            raise MissingJITCacheError(
                "JIT compilation is disabled via MATE_DISABLE_JIT, "
                "but no matching AOT module exists.",
                spec=self,
            )

        if need_lock:
            with FileLock(str(self.lock_path)):
                self.write_ninja()
                run_ninja(self.build_dir, self.ninja_path, verbose)
        else:
            self.write_ninja()
            run_ninja(self.build_dir, self.ninja_path, verbose)
        return self.jit_library_path

    def build_and_load(self):
        if self.is_aot:
            return self.load(self.aot_path)  # type: ignore[arg-type]

        verbose = os.environ.get("MATE_JIT_VERBOSE", "0") == "1"
        with FileLock(str(self.lock_path)):
            library_path = self.build(verbose=verbose, need_lock=False)
            return self.load(library_path)


def gen_jit_spec(
    name: str,
    sources: Sequence[Path | str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[Sequence[Path | str]] = None,
    generated_sources: Optional[Mapping[Path | str, str]] = None,
    aot_path: Optional[Path | str] = None,
) -> JitSpec:
    normalized_sources = [Path(path) for path in sources]
    normalized_generated_sources = (
        {Path(path): content for path, content in generated_sources.items()}
        if generated_sources is not None
        else {}
    )
    for generated_path in normalized_generated_sources:
        if generated_path not in normalized_sources:
            normalized_sources.append(generated_path)

    spec = JitSpec(
        name=name,
        sources=normalized_sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_dirs=(
            [Path(path) for path in extra_include_paths]
            if extra_include_paths is not None
            else None
        ),
        generated_sources=normalized_generated_sources,
        aot_path=Path(aot_path) if aot_path is not None else default_aot_path(name),
    )
    jit_spec_registry.register(spec)
    return spec


def _dedupe_specs(specs: Sequence[JitSpec]) -> List[JitSpec]:
    deduped: Dict[str, JitSpec] = {}
    for spec in specs:
        existing = deduped.get(spec.name)
        if existing is None:
            deduped[spec.name] = spec
        elif existing != spec:
            raise ValueError(f"JIT spec collision for name={spec.name}")
    return list(deduped.values())


@contextmanager
def temporary_max_jobs(jobs: Optional[int]):
    if jobs is None:
        yield
        return

    previous = os.environ.get("MAX_JOBS")
    os.environ["MAX_JOBS"] = str(jobs)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MAX_JOBS", None)
        else:
            os.environ["MAX_JOBS"] = previous


def build_jit_specs(
    specs: Sequence[JitSpec],
    *,
    jobs: Optional[int] = None,
    verbose: bool = False,
    skip_prebuilt: bool = True,
) -> None:
    unique_specs = _dedupe_specs(specs)
    lines: List[str] = []
    for spec in unique_specs:
        if skip_prebuilt and spec.is_aot:
            continue
        with FileLock(str(spec.lock_path)):
            spec.write_ninja()
        subninja_path = spec.ninja_path.resolve().as_posix().replace(":", "$:")
        lines.append(f"subninja {subninja_path}")

    if not lines:
        return

    tmpdir = get_tmpdir()
    tmpdir.mkdir(parents=True, exist_ok=True)
    top_level_ninja = tmpdir / "mate_jit.ninja"
    with FileLock(str(tmpdir / "mate_jit.lock")), temporary_max_jobs(jobs):
        write_if_different(
            top_level_ninja,
            "\n".join(["ninja_required_version = 1.3", *lines, ""]),
        )
        run_ninja(jit_env.MATE_JIT_DIR, top_level_ninja, verbose)


def copy_built_kernels(
    specs: Sequence[JitSpec],
    out_dir: Path,
) -> None:
    unique_specs = _dedupe_specs(specs)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    for spec in unique_specs:
        destination = spec.aot_path or default_aot_path(spec.name, out_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.jit_library_path, destination)


def clear_cache_dir() -> None:
    if jit_env.MATE_JIT_DIR.exists():
        shutil.rmtree(jit_env.MATE_JIT_DIR)
