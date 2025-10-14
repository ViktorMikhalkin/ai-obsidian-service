# src/ai_obsidian_service/__init__.py
"""
AI Obsidian Service - public package API.

Minimal initialization only:
- package version (__version__)
- quiet base logger
"""

from __future__ import annotations

import logging

# Do not attach any real handlers here; keep logging quiet by default.
logging.getLogger(__name__).addHandler(logging.NullHandler())

# Package version (release-please updates this file during release).
try:
    from .__version__ import __version__
except Exception:  # pragma: no cover
    __version__ = "0.1.3"  # fallback if version file is missing during build

# Note: Removed problematic re-exports until the indexer module is properly implemented
# TODO: Add back build_index and get_index_status when they're implemented in the indexer module

__all__ = ["__version__"]
