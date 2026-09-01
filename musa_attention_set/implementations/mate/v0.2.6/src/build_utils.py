"""Build utilities for version management and git operations."""

import subprocess
from pathlib import Path
from typing import Optional


def get_git_version(cwd: Optional[Path] = None) -> str:
    """Get git commit hash (full).

    Args:
        cwd: Working directory for git command. If None, uses current directory.

    Returns:
        Git commit hash or "unknown" if git is not available.
    """
    try:
        git_version = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return git_version
    except Exception:
        return "unknown"


def get_git_short_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Get short git commit hash (7 characters).

    Args:
        cwd: Working directory for git command. If None, uses current directory.

    Returns:
        Short commit hash (e.g., "g40c8139") or None if git is not available.
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=cwd,
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        return f"g{commit}"
    except Exception:
        return None


def build_version_string(
    base_version: str,
    cwd: Optional[Path] = None,
    dev_suffix: str = "",
    local_version: Optional[str] = None,
) -> tuple[str, str]:
    """Build full package version string (git version separate).

    Args:
        base_version: Base version from version.txt (e.g., "0.1.2")
        cwd: Working directory for git commands
        dev_suffix: Optional dev release suffix (e.g., "1" for .dev1)
        local_version: Optional explicit local version string

    Returns:
        Tuple of (full_version, git_version)
        - full_version: e.g., "0.1.2" or "0.1.2+nightly"
        - git_version: full git commit hash for CLI display
    """
    version = base_version

    # Add dev suffix if specified
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version (full hash for metadata/CLI display)
    git_version = get_git_version(cwd=cwd)

    if local_version:
        version = f"{version}+{local_version}"

    return version, git_version
