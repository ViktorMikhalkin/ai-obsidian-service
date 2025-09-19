"""
Public facade for the 'indexer' subsystem.

Provides thin wrappers:
- build_index(...)
- get_index_status(...)

Adjust imported function names below if they differ in your codebase
(e.g., rename to `.build: main` or `.status: main`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .build import main as _build_index
    from .status import main as _get_index_status
else:
    try:
        from .build import main as _build_index  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        _build_index = None

    try:
        from .status import main as _get_index_status  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        _get_index_status = None


def build_index(*args: Any, **kwargs: Any) -> Any:
    if _build_index is None:
        raise RuntimeError("build_index is unavailable: import error in indexer.build")
    return _build_index(*args, **kwargs)


def get_index_status(*args: Any, **kwargs: Any) -> Any:
    if _get_index_status is None:
        raise RuntimeError(
            "get_index_status is unavailable: import error in indexer.status"
        )
    return _get_index_status(*args, **kwargs)


__all__ = ["build_index", "get_index_status"]
