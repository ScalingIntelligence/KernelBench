#!/usr/bin/env python3
"""Validate that documented public Python APIs match the intended code surface.

The manifest lives in ``docs/api_manifest.yaml``. It intentionally uses the
JSON subset of YAML so this checker can stay stdlib-only.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SUMMARY_BULLET_RE = re.compile(r"^\s*-\s+``(mate(?:\.[A-Za-z_]\w*)+)``")
CURRENTMODULE_RE = re.compile(r"^\s*\.\.\s+currentmodule::\s+([A-Za-z_][\w.]*)\s*$")
AUTOFUNCTION_RE = re.compile(r"^\s*\.\.\s+autofunction::\s+([A-Za-z_][\w.]*)\s*$")
PY_FUNCTION_RE = re.compile(r"^\s*\.\.\s+py:function::\s+([A-Za-z_][\w.]*)")
MARKDOWN_CODE_RE = re.compile(r"mate(?:\.[A-Za-z_]\w*)+")


@dataclass(frozen=True)
class SourceSpec:
    kind: str
    path: str
    module_targets: tuple[str, ...] = ()
    import_module: str | None = None
    only_names: tuple[str, ...] = ()
    exclude_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class PageSpec:
    path: str
    extractor: str
    sources: tuple[SourceSpec, ...]


@dataclass(frozen=True)
class GroupSpec:
    name: str
    summary_sources: tuple[SourceSpec, ...]
    detail_pages: tuple[PageSpec, ...]


def _load_manifest(manifest_path: Path) -> tuple[str, tuple[GroupSpec, ...]]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    groups: list[GroupSpec] = []
    for raw_group in data["groups"]:
        groups.append(
            GroupSpec(
                name=raw_group["name"],
                summary_sources=tuple(
                    _build_source_spec(item) for item in raw_group["summary_sources"]
                ),
                detail_pages=tuple(
                    _build_page_spec(item) for item in raw_group.get("detail_pages", [])
                ),
            )
        )

    return data["summary_page"], tuple(groups)


def _build_source_spec(data: dict[str, object]) -> SourceSpec:
    return SourceSpec(
        kind=str(data["kind"]),
        path=str(data["path"]),
        module_targets=tuple(str(item) for item in data.get("module_targets", [])),
        import_module=str(data["import_module"]) if "import_module" in data else None,
        only_names=tuple(str(item) for item in data.get("only_names", [])),
        exclude_names=tuple(str(item) for item in data.get("exclude_names", [])),
    )


def _build_page_spec(data: dict[str, object]) -> PageSpec:
    return PageSpec(
        path=str(data["path"]),
        extractor=str(data.get("extractor", "rst_api")),
        sources=tuple(_build_source_spec(item) for item in data["sources"]),
    )


def _module_name_from_path(path: str) -> str:
    if not path.endswith(".py"):
        raise ValueError(f"expected a Python module path, got {path!r}")
    return path[:-3].replace("/", ".")


def _read_tree(root: Path, relative_path: str) -> ast.Module:
    source = (root / relative_path).read_text(encoding="utf-8")
    return ast.parse(source, filename=relative_path)


def _collect_summary_symbols(root: Path, summary_page: str) -> set[str]:
    symbols: set[str] = set()
    for line in (root / summary_page).read_text(encoding="utf-8").splitlines():
        match = SUMMARY_BULLET_RE.search(line)
        if match:
            symbols.add(match.group(1))
    return symbols


def _collect_page_symbols(root: Path, page: PageSpec) -> set[str]:
    if page.extractor == "markdown_code":
        content = (root / page.path).read_text(encoding="utf-8")
        return set(MARKDOWN_CODE_RE.findall(content))
    if page.extractor != "rst_api":
        raise ValueError(f"unsupported page extractor {page.extractor!r}")
    return _collect_rst_page_symbols(root / page.path)


def _collect_rst_page_symbols(path: Path) -> set[str]:
    symbols: set[str] = set()
    current_module: str | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        match = CURRENTMODULE_RE.match(line)
        if match:
            current_module = match.group(1)
            continue

        match = AUTOFUNCTION_RE.match(line)
        if match:
            name = match.group(1)
            if "." in name:
                symbols.add(name)
            elif current_module is not None:
                symbols.add(f"{current_module}.{name}")
            else:
                raise ValueError(
                    f"autofunction {name!r} in {path} has no currentmodule context"
                )
            continue

        match = PY_FUNCTION_RE.match(line)
        if match:
            symbols.add(match.group(1))

    return symbols


def _collect_source_symbols(root: Path, spec: SourceSpec) -> set[str]:
    if spec.kind == "lazy_attr_modules":
        names = _collect_lazy_attr_names(root, spec.path, set(spec.module_targets))
        return _apply_filters({f"mate.{name}" for name in names}, spec)
    if spec.kind == "module_mate_api":
        module_name = _module_name_from_path(spec.path)
        names = _collect_mate_api_names(root, spec.path)
        return _apply_filters({f"{module_name}.{name}" for name in names}, spec)
    if spec.kind == "module_import_from":
        module_name = _module_name_from_path(spec.path)
        names = _collect_imported_names(root, spec.path, spec.import_module)
        return _apply_filters({f"{module_name}.{name}" for name in names}, spec)
    raise ValueError(f"unsupported source kind {spec.kind!r}")


def _apply_filters(symbols: set[str], spec: SourceSpec) -> set[str]:
    if spec.only_names:
        allowed = set(spec.only_names)
        symbols = {symbol for symbol in symbols if symbol.rsplit(".", 1)[-1] in allowed}
    if spec.exclude_names:
        blocked = set(spec.exclude_names)
        symbols = {
            symbol for symbol in symbols if symbol.rsplit(".", 1)[-1] not in blocked
        }
    return symbols


def _collect_lazy_attr_names(
    root: Path, relative_path: str, module_targets: set[str]
) -> set[str]:
    tree = _read_tree(root, relative_path)
    lazy_mapping: dict[str, str] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_LAZY_ATTR_MODULES":
                value = ast.literal_eval(node.value)
                if not isinstance(value, dict):
                    raise ValueError("_LAZY_ATTR_MODULES must be a dict literal")
                lazy_mapping = {
                    str(name): str(module_name) for name, module_name in value.items()
                }
                break

    return {
        name
        for name, module_name in lazy_mapping.items()
        if module_name in module_targets
    }


def _decorator_name(decorator: ast.expr) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    return None


def _collect_mate_api_names(root: Path, relative_path: str) -> set[str]:
    tree = _read_tree(root, relative_path)
    names: set[str] = set()

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            _decorator_name(decorator) == "mate_api"
            for decorator in node.decorator_list
        ):
            names.add(node.name)

    return names


def _collect_imported_names(
    root: Path, relative_path: str, import_module: str | None
) -> set[str]:
    if import_module is None:
        raise ValueError("module_import_from sources require import_module")
    tree = _read_tree(root, relative_path)
    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == import_module:
            for alias in node.names:
                if alias.name == "*":
                    raise ValueError(
                        f"wildcard import from {import_module!r} is not supported in {relative_path}"
                    )
                names.add(alias.asname or alias.name)

    return names


def _collect_expected_symbols(specs: Iterable[SourceSpec], root: Path) -> set[str]:
    symbols: set[str] = set()
    for spec in specs:
        symbols.update(_collect_source_symbols(root, spec))
    return symbols


def run_check(root: Path, manifest_path: Path) -> list[str]:
    summary_page, groups = _load_manifest(manifest_path)
    summary_symbols = _collect_summary_symbols(root, summary_page)

    errors: list[str] = []
    expected_summary_symbols: set[str] = set()

    for group in groups:
        group_symbols = _collect_expected_symbols(group.summary_sources, root)
        expected_summary_symbols.update(group_symbols)

        missing_summary = sorted(group_symbols - summary_symbols)
        if missing_summary:
            errors.append(
                f"[{group.name}] summary page {summary_page} is missing: "
                + ", ".join(missing_summary)
            )

        for page in group.detail_pages:
            expected_page_symbols = _collect_expected_symbols(page.sources, root)
            actual_page_symbols = _collect_page_symbols(root, page)

            missing_page = sorted(expected_page_symbols - actual_page_symbols)
            unexpected_page = sorted(actual_page_symbols - expected_page_symbols)

            if missing_page:
                errors.append(
                    f"[{group.name}] detail page {page.path} is missing: "
                    + ", ".join(missing_page)
                )
            if unexpected_page:
                errors.append(
                    f"[{group.name}] detail page {page.path} has stale or unowned symbols: "
                    + ", ".join(unexpected_page)
                )

    unexpected_summary = sorted(summary_symbols - expected_summary_symbols)
    if unexpected_summary:
        errors.append(
            f"[summary] {summary_page} has stale or unowned symbols: "
            + ", ".join(unexpected_summary)
        )

    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=".",
        help="repository root to check (default: current directory)",
    )
    parser.add_argument(
        "--manifest",
        default="docs/api_manifest.yaml",
        help="manifest path relative to --root (default: docs/api_manifest.yaml)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(args.root).resolve()
    manifest_path = root / args.manifest

    errors = run_check(root, manifest_path)
    if errors:
        print("API docs check failed.", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    summary_page, groups = _load_manifest(manifest_path)
    expected_count = len(
        set().union(
            *(
                _collect_expected_symbols(group.summary_sources, root)
                for group in groups
            )
        )
    )
    print(
        f"API docs check passed: {expected_count} summary symbols verified in {summary_page}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
