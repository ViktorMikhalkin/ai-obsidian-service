# src/ai_obsidian_service/__init__.py
"""
AI Obsidian Service - public package API.

Minimal initialization only:
- package version (__version__)
- quiet base logger
- lightweight re-exports of common operations (if available)
"""

from __future__ import annotations

import logging

# Do not attach any real handlers here; keep logging quiet by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package version (release-please updates this file during release).
try:
    from .__version__ import __version__
except Exception:  # pragma: no cover
    __version__ = "0.0.0"  # fallback if version file is missing during build

# Lightweight re-exports from subpackages. Import only symbols that are cheap and safe.
try:
    from .indexer import build_index, get_index_status
except Exception:
    # Keep the package importable even if optional deps are not installed.
    pass

__all__ = ["__version__", "build_index", "get_index_status"]
