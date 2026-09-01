from __future__ import annotations

from collections.abc import Iterable


def resolve_backend(
    backend: str | None,
    *,
    supported: Iterable[str],
    allow_auto: bool = True,
    default: str | None = None,
) -> str:
    if backend is None:
        if default is None:
            raise ValueError("backend must be provided")
        backend = default

    choices = tuple(dict.fromkeys(supported))
    if backend == "auto":
        if not allow_auto:
            raise ValueError(f"backend must be one of {list(choices)}")
        return backend
    if backend not in choices:
        allowed = ["auto", *choices] if allow_auto else list(choices)
        raise ValueError(f"backend must be one of {allowed}")
    return backend
