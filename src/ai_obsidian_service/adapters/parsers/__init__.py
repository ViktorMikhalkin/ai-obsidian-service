from __future__ import annotations

# Single source of truth for ports/protocols:
from ai_obsidian_service.core import DocumentParser  # re-export from ports.interfaces

from .epub_parser import EpubParser

# Import concrete parser implementations
from .md_parser import MarkdownParser
from .pdf_parser import PdfParser

_REGISTRY: list[type[DocumentParser]] = [MarkdownParser, PdfParser, EpubParser]

def register_parser(cls: type[DocumentParser]) -> None:
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)

def build_parsers() -> list[DocumentParser]:
    # Instantiate all registered parsers; "available" gating can be added per class later
    return [cls() for cls in _REGISTRY]

# Back-compat export: historically code pulled default_parsers as a value.
# Keep callable semantic: default_parsers() -> list[DocumentParser]
def default_parsers() -> list[DocumentParser]:
    return build_parsers()

__all__ = [
    "DocumentParser",
    "MarkdownParser",
    "PdfParser",
    "EpubParser",
    "register_parser",
    "build_parsers",
    "default_parsers",
]
