#!/usr/bin/env python3
"""Validate repository-local agent skills and their dependency direction."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MARKDOWN_LINK_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<(?P<bracketed>[^>]+)>|(?P<plain>[^)\s]+))"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
H2_RE = re.compile(r"^##\s+(?P<title>.+?)\s*#*\s*$")
IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "3rdparty",
    "build",
    "dist",
    "venv",
}
MAX_SKILL_LINES = 500


@dataclass(frozen=True)
class Skill:
    path: Path
    relative_path: str
    name: str
    description: str
    body_start_line: int
    is_top_level: bool


@dataclass(frozen=True)
class ValidationResult:
    skills: tuple[Skill, ...]
    errors: tuple[str, ...]


def _is_skill_file(path: Path, repo_root: Path) -> bool:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return False
    parts = relative.parts
    return (
        len(parts) >= 3
        and parts[-1] == "SKILL.md"
        and parts[-3] == "skills"
        and not any(part in IGNORED_DIRECTORY_NAMES for part in parts[:-3])
    )


def _discover_skill_files(repo_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in repo_root.rglob("SKILL.md")
            if _is_skill_file(path, repo_root)
        ),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )


def _parse_skill(path: Path, repo_root: Path) -> tuple[Skill | None, list[str]]:
    relative = path.relative_to(repo_root).as_posix()
    errors: list[str] = []
    if path.is_symlink():
        return None, [f"{relative}: SKILL.md must not be a symbolic link"]

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return None, [f"{relative}: cannot read SKILL.md as UTF-8: {exc}"]

    lines = text.splitlines()
    if len(lines) > MAX_SKILL_LINES:
        errors.append(
            f"{relative}: SKILL.md has {len(lines)} lines; maximum is {MAX_SKILL_LINES}"
        )
    if not lines or lines[0].strip() != "---":
        return None, [*errors, f"{relative}: missing opening YAML frontmatter"]

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration:
        return None, [*errors, f"{relative}: missing closing YAML frontmatter"]

    try:
        frontmatter = yaml.safe_load("\n".join(lines[1:closing_index]))
    except yaml.YAMLError as exc:
        return None, [*errors, f"{relative}: invalid YAML frontmatter: {exc}"]

    if not isinstance(frontmatter, dict):
        return None, [*errors, f"{relative}: frontmatter must be a mapping"]
    keys = set(frontmatter)
    if keys != {"name", "description"}:
        errors.append(f"{relative}: frontmatter must contain only name and description")

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip():
        errors.append(f"{relative}: name must be a non-empty string")
        name = ""
    else:
        name = name.strip()
        if len(name) >= 64 or not SKILL_NAME_RE.fullmatch(name):
            errors.append(
                f"{relative}: name must be lowercase kebab-case and under 64 characters"
            )
        if path.parent.name != name:
            errors.append(f"{relative}: directory name must match skill name {name!r}")

    if not isinstance(description, str) or not description.strip():
        errors.append(f"{relative}: description must be a non-empty string")
        description = ""
    else:
        description = description.strip()

    if not "\n".join(lines[closing_index + 1 :]).strip():
        errors.append(f"{relative}: skill body must not be empty")

    relative_parts = path.relative_to(repo_root).parts
    skill = Skill(
        path=path,
        relative_path=relative,
        name=name,
        description=description,
        body_start_line=closing_index + 2,
        is_top_level=len(relative_parts) == 3,
    )
    return skill, errors


def _target_skill_layer(path: Path, repo_root: Path) -> bool | None:
    if not _is_skill_file(path, repo_root):
        return None
    return len(path.relative_to(repo_root).parts) == 3


def _validate_links(skill: Skill, repo_root: Path) -> list[str]:
    errors: list[str] = []
    text = skill.path.read_text(encoding="utf-8")
    current_h2: str | None = None
    first_h2: str | None = None
    has_public_baseline_dependency = False
    repo_root_resolved = repo_root.resolve()
    fence_marker: str | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence_marker is None:
                fence_marker = marker
            elif marker == fence_marker:
                fence_marker = None
            continue
        if fence_marker is not None:
            continue

        heading = H2_RE.match(line)
        if heading:
            current_h2 = heading.group("title").strip()
            if first_h2 is None and line_number >= skill.body_start_line:
                first_h2 = current_h2

        for match in MARKDOWN_LINK_RE.finditer(line):
            target = match.group("bracketed") or match.group("plain")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if parsed.path.startswith("/"):
                errors.append(
                    f"{skill.relative_path}:{line_number}: local link must be relative"
                )
                continue

            decoded_path = unquote(parsed.path)
            candidate = (skill.path.parent / decoded_path).resolve()
            try:
                candidate.relative_to(repo_root_resolved)
            except ValueError:
                errors.append(
                    f"{skill.relative_path}:{line_number}: local link escapes "
                    "the repository"
                )
                continue
            if not candidate.exists():
                errors.append(
                    f"{skill.relative_path}:{line_number}: local link target "
                    f"does not exist: {decoded_path}"
                )
                continue

            target_is_top_level = _target_skill_layer(candidate, repo_root_resolved)
            if target_is_top_level is None or candidate == skill.path.resolve():
                continue
            if (
                not skill.is_top_level
                and target_is_top_level
                and current_h2 == "Public Baseline"
            ):
                has_public_baseline_dependency = True
                continue
            errors.append(
                f"{skill.relative_path}:{line_number}: cross-skill dependencies "
                "are allowed only from a nested skill's Public Baseline section "
                "to a top-level public skill"
            )

    if has_public_baseline_dependency and first_h2 != "Public Baseline":
        errors.append(
            f"{skill.relative_path}: Public Baseline must be the first level-two "
            "section when declaring a public skill dependency"
        )
    return errors


def validate_repository(repo_root: Path) -> ValidationResult:
    repo_root = repo_root.resolve()
    skills: list[Skill] = []
    errors: list[str] = []

    for path in _discover_skill_files(repo_root):
        skill, parse_errors = _parse_skill(path, repo_root)
        errors.extend(parse_errors)
        if skill is not None:
            skills.append(skill)

    skills_by_name: dict[str, list[Skill]] = {}
    for skill in skills:
        if skill.name:
            skills_by_name.setdefault(skill.name, []).append(skill)
    for name, matching_skills in sorted(skills_by_name.items()):
        if len(matching_skills) > 1:
            paths = ", ".join(skill.relative_path for skill in matching_skills)
            errors.append(f"duplicate skill name {name!r}: {paths}")

    for skill in skills:
        errors.extend(_validate_links(skill, repo_root))

    return ValidationResult(tuple(skills), tuple(errors))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate all repository-local agent skills."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan (default: parent of scripts/)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_repository(args.repo_root)
    if result.errors:
        for error in result.errors:
            print(f"error: {error}", file=sys.stderr)
        print(
            f"Skill validation failed with {len(result.errors)} error(s).",
            file=sys.stderr,
        )
        return 1
    print(f"Validated {len(result.skills)} repository skill(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
