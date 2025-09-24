
from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from ai_obsidian_service.core import DocId, Document, DocumentParser

__all__ = ["parse_markdown", "MarkdownParser"]

# Regex for ATX-style headers (#, ##, ### ... ######)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def parse_markdown(text: str) -> list[tuple[str, str]]:
    """Split Markdown text into sections by ATX headers.

    Returns list of (header, body). If no headers, returns [('', text)].
    """
    if not text:
        return [("", "")]
    positions: list[tuple[int, str]] = []
    for m in _HEADER_RE.finditer(text):
        positions.append((m.start(), m.group(0)))
    if not positions:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    for i, (start, hdr) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[start:end]
        sections.append((hdr, body))
    return sections


class MarkdownParser(DocumentParser):
    _EXTS = {".md", ".markdown"}

    def can_parse(self, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in self._EXTS
        except Exception:
            return False

    def parse_text(self, text: str, *, source: str = "<memory>") -> Document:
        mime = "text/markdown"
        return Document(id=DocId(source), path=source, mime=mime, text=text)

    def parse(self, path: str) -> Document:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        mime = mimetypes.guess_type(str(p))[0] or "text/markdown"
        return Document(id=DocId(str(p)), path=str(p), mime=mime, text=text)
