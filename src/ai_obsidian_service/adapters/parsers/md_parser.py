from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from ai_obsidian_service.core import DocId, Document, DocumentParser

__all__ = ["parse_markdown", "MarkdownParser"]

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

def parse_markdown(text: str) -> list[tuple[str, str]]:
    """Split Markdown into sections by ATX headers.
    Returns a list of (title, body) tuples:
    - title: header text (empty string for preface before first header)
    - body: content under that header, stripped of leading/trailing whitespace
    """
    sections: list[tuple[str, str]] = []
    title: str | None = None
    body_lines: list[str] = []
    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            if title is not None or body_lines:
                sections.append((title or "", "\n".join(body_lines).strip()))
                body_lines = []
            title = m.group(2).strip()
        else:
            body_lines.append(line)
    if title is None and not body_lines:
        return [("", "")]
    sections.append((title or "", "\n".join(body_lines).strip()))
    return sections

class MarkdownParser(DocumentParser):
    """Markdown parser that produces a Document with raw text.
    Sectioning helper is exposed via parse_markdown(), but the adapter fulfils the
    canonical port: can_parse(path)->bool, parse(path)->Document.
    """

    _EXTS = {".md", ".markdown"}

    def can_parse(self, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in self._EXTS
        except Exception:
            return False

    def parse(self, path: str) -> Document:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        mime = mimetypes.guess_type(str(p))[0] or "text/markdown"
        return Document(id=DocId(str(p)), path=str(p), mime=mime, text=text)
