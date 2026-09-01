#!/usr/bin/env python3
"""Validate provenance metadata and SHA-256 manifests for archived sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = ROOT / "implementations"
MANIFEST = ROOT / "manifest.csv"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_METADATA = {
    "project",
    "repository",
    "revision",
    "revision_label",
    "retrieved_at",
    "license",
    "operator_families",
    "source_scope",
    "archive_status",
    "validation_status",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archived_files(snapshot: Path) -> list[Path]:
    candidates = [snapshot / "LICENSE.upstream"]
    candidates.extend((snapshot / "src").rglob("*"))
    return sorted(
        (path for path in candidates if path.is_file()),
        key=lambda path: path.relative_to(snapshot).as_posix(),
    )


def expected_sums(snapshot: Path) -> list[str]:
    return [
        f"{sha256(path)}  {path.relative_to(snapshot).as_posix()}"
        for path in archived_files(snapshot)
    ]


def load_manifest() -> dict[tuple[str, str], dict[str, str]]:
    with MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row["project"], row["revision"])
        if key in result:
            raise ValueError(f"duplicate manifest key: {key}")
        result[key] = row
    return result


def verify_snapshot(
    snapshot: Path, manifest: dict[tuple[str, str], dict[str, str]], write: bool
) -> list[str]:
    errors: list[str] = []
    metadata_path = snapshot / "metadata.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{metadata_path}: {exc}"]

    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        errors.append(f"{metadata_path}: missing fields {sorted(missing)}")
    revision = metadata.get("revision", "")
    if not SHA_RE.fullmatch(revision):
        errors.append(f"{metadata_path}: revision is not a full 40-char SHA")
    if not (snapshot / "LICENSE.upstream").is_file():
        errors.append(f"{snapshot}: missing LICENSE.upstream")
    if not (snapshot / "src").is_dir():
        errors.append(f"{snapshot}: missing src directory")

    forbidden = [
        path
        for path in snapshot.rglob("*")
        if path.name in {".git", "AGENTS.md", "CLAUDE.md"}
    ]
    if forbidden:
        errors.append(
            f"{snapshot}: forbidden repository/instruction files: "
            + ", ".join(str(path.relative_to(snapshot)) for path in forbidden)
        )

    key = (metadata.get("project", ""), revision)
    row = manifest.get(key)
    if row is None:
        errors.append(f"{metadata_path}: no matching manifest.csv row")
    else:
        expected_path = snapshot.relative_to(ROOT).as_posix()
        if row["snapshot"] != expected_path:
            errors.append(
                f"{metadata_path}: manifest snapshot is {row['snapshot']!r}, "
                f"expected {expected_path!r}"
            )

    lines = expected_sums(snapshot)
    sums_path = snapshot / "SHA256SUMS"
    if write:
        sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    elif not sums_path.is_file():
        errors.append(f"{snapshot}: missing SHA256SUMS")
    else:
        recorded = sums_path.read_text(encoding="utf-8").splitlines()
        if recorded != lines:
            errors.append(f"{snapshot}: SHA256SUMS does not match archived files")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write", action="store_true", help="regenerate every SHA256SUMS file"
    )
    args = parser.parse_args()

    try:
        manifest = load_manifest()
    except (OSError, ValueError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1

    metadata_files = sorted(IMPLEMENTATIONS.glob("*/*/metadata.json"))
    errors: list[str] = []
    found_keys: set[tuple[str, str]] = set()
    for metadata_path in metadata_files:
        snapshot = metadata_path.parent
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        found_keys.add((metadata.get("project", ""), metadata.get("revision", "")))
        errors.extend(verify_snapshot(snapshot, manifest, args.write))

    missing_snapshots = set(manifest) - found_keys
    for key in sorted(missing_snapshots):
        errors.append(f"manifest row has no metadata.json snapshot: {key}")

    if errors:
        print("Snapshot verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    action = "wrote" if args.write else "verified"
    print(f"{action} {len(metadata_files)} snapshots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
