from __future__ import annotations

from typing import cast, List

from ai_obsidian_service.adapters.parsers.epub_parser import EpubParser
from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.parsers.pdf_parser import PdfParser
from ai_obsidian_service.ports.interfaces import DocumentParser


def all_parsers() -> List[DocumentParser]:
    raw = [MarkdownParser(), PdfParser(), EpubParser()]
    return cast(List[DocumentParser], cast(object, raw))
