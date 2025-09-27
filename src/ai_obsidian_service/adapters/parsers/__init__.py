from __future__ import annotations

from typing import List

from ai_obsidian_service.adapters.parsers.epub_parser import EpubParser
from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.parsers.pdf_parser import PdfParser


def default_parsers() -> list[object]:
    return [MarkdownParser(), PdfParser(), EpubParser()]

all_parsers = default_parsers
