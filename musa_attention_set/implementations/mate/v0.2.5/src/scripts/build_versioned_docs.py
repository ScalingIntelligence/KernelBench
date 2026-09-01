#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSIONS_PATH = ROOT / "docs" / "versions.json"
OVERLAY_FILES = [
    Path("docs/source/conf.py"),
    Path("docs/source/_static/custom.css"),
    Path("docs/source/_static/mermaid-init.js"),
    Path("docs/source/_templates/sidebar/scroll-start.html"),
    Path("docs/source/_templates/sidebar/scroll-end.html"),
    Path("docs/versions.json"),
]


@dataclass
class VersionBuild:
    version: str
    ref: str | None = None
    aliases: list[str] = field(default_factory=list)
    overlay_current_ui: bool = False

    @property
    def targets(self) -> list[str]:
        return [self.version, *self.aliases]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the MATE docs site for all configured versions."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "build" / "versioned-html",
        help="Directory that will receive the versioned static site.",
    )
    return parser.parse_args()


def load_version_builds() -> tuple[str, list[VersionBuild]]:
    with VERSIONS_PATH.open(encoding="utf-8") as versions_file:
        versions_data = json.load(versions_file)

    latest_release = str(versions_data.get("latest", "")).strip()
    builds = []
    seen_versions = set()

    for raw_entry in versions_data.get("versions", []):
        if isinstance(raw_entry, str):
            build = VersionBuild(version=raw_entry.strip())
        else:
            build = VersionBuild(
                version=str(raw_entry.get("version", "")).strip(),
                ref=raw_entry.get("ref"),
                aliases=list(raw_entry.get("aliases", [])),
                overlay_current_ui=bool(raw_entry.get("overlay_current_ui", False)),
            )

        if not build.version or build.version in seen_versions:
            continue

        seen_versions.add(build.version)
        builds.append(build)

    if latest_release and latest_release not in seen_versions:
        builds.insert(0, VersionBuild(version=latest_release, aliases=["latest"]))

    for build in builds:
        if build.version == latest_release and "latest" not in build.aliases:
            build.aliases.insert(0, "latest")

    return latest_release, builds


def export_ref(ref: str, destination: Path) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", ref],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(destination)


def apply_current_ui_overlay(destination: Path) -> None:
    for relative_path in OVERLAY_FILES:
        source_path = ROOT / relative_path
        target_path = destination / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def build_docs(
    source_root: Path,
    output_dir: Path,
    docs_release: str,
    docs_current_version: str,
    latest_release: str,
) -> None:
    doctrees_dir = output_dir.parent / f".doctrees-{docs_current_version}"
    env = os.environ.copy()
    env["DOCS_RELEASE"] = docs_release
    env["DOCS_CURRENT_VERSION"] = docs_current_version
    env["DOCS_LATEST_VERSION"] = latest_release
    env.setdefault("MATE_MUSA_ARCH_LIST", "3.1")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    doctrees_dir.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-b",
            "html",
            "-d",
            str(doctrees_dir),
            str(source_root / "docs" / "source"),
            str(output_dir),
        ],
        cwd=source_root,
        env=env,
        check=True,
    )


def write_root_redirect(output_root: Path) -> None:
    (output_root / "index.html").write_text(
        """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=latest/">
    <title>MATE Docs Redirect</title>
  </head>
  <body>
    <p>Redirecting to <a href="latest/">latest/</a>…</p>
  </body>
</html>
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    latest_release, builds = load_version_builds()
    output_root = args.output.resolve()

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mate-versioned-docs-") as temp_root_str:
        temp_root = Path(temp_root_str)
        for build in builds:
            if build.ref:
                source_root = temp_root / build.version
                export_ref(build.ref, source_root)
                if build.overlay_current_ui:
                    apply_current_ui_overlay(source_root)
            else:
                source_root = ROOT

            for target in build.targets:
                build_docs(
                    source_root=source_root,
                    output_dir=output_root / target,
                    docs_release=build.version,
                    docs_current_version=target,
                    latest_release=latest_release,
                )

    write_root_redirect(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
