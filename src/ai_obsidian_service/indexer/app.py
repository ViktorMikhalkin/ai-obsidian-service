"""Compatibility shim: re-export FastAPI app from the new API module."""

from ai_obsidian_service.api.app import app  # re-export

__all__ = ["app"]
