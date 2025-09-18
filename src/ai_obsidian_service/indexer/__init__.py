# src/ai_obsidian_service/indexer/__init__.py
"""
Public facade for the 'indexer' subsystem.

Provides thin wrappers:
- build_index(...)
- get_index_status(...)

Adjust imported function names below if they differ in your codebase
(e.g., rename to `.build: main` or `.status: status`).
"""

from __future__ import annotations

# build_index wrapper
try:
    # Replace `build_index` with the actual function name if different.
    from .build import build_index as _build_index  # noqa: F401
except Exception:
    _build_index = None  # type: ignore[misc,assignment]


def build_index(*args, **kwargs):
    """
    Thin wrapper for the index build entry point.
    Delegates to ai_obsidian_service.indexer.build.build_index (or equivalent).
    """
    if _build_index is None:
        raise RuntimeError(
            "build_index is unavailable: optional dependencies not installed or import error in indexer.build"
        )
    return _build_index(*args, **kwargs)


# get_index_status wrapper
try:
    # Replace `get_index_status` with the actual function name if different.
    from .status import get_index_status as _get_index_status  # noqa: F401
except Exception:
    _get_index_status = None  # type: ignore[misc,assignment]


def get_index_status(*args, **kwargs):
    """
    Thin wrapper for the index status entry point.
    Delegates to ai_obsidian_service.indexer.status.get_index_status (or equivalent).
    """
    if _get_index_status is None:
        raise RuntimeError(
            "get_index_status is unavailable: optional dependencies not installed or import error in indexer.status"
        )
    return _get_index_status(*args, **kwargs)


__all__ = ["build_index", "get_index_status"]
