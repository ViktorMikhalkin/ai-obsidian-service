from __future__ import annotations

import mimetypes
import re
from pathlib import Path

from ai_obsidian_service.core import DocId, Document, DocumentParser

__all__ = ["parse_markdown", "MarkdownParser"]

# Regex for ATX-style headers (#, ##, ### ... ######)
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(text: str) -> list[tuple[str, str]]:
    """Split Markdown text into (header, body) sections.

    Indexing-oriented behavior:
    - Each ATX header starts a new section (title = header text).
    - Text before the first header becomes a section with an empty title "".
    - Orphan blocks (non-header text appearing after a blank line) become
      separate sections with an empty title "".
    - Back-to-back headers still create sections with empty bodies.

    Returns:
        list[(title, body)]
    """
    sections: list[tuple[str, str]] = []
    title: str | None = None
    buf: list[str] = []
    prev_blank = False

    def flush_any() -> None:
        """Append the current (title, body) section even if body is empty,
        as long as a title exists OR we collected some body lines.
        This ensures back-to-back headers yield empty sections.
        """
        nonlocal title, buf
        if title is not None or buf:
            body = "\n".join(buf).strip()
            sections.append((title or "", body))
            buf = []

    for line in text.splitlines():
        m = _HEADER_RE.match(line)
        if m:
            # New header → close previous section (empty body allowed)
            flush_any()
            title = m.group(2).strip()
            prev_blank = False
            continue

        # Orphan-block heuristic:
        # If we have already collected some body lines and the previous line
        # was blank, and the current line is non-header text, start a new
        # unnamed section. This applies BOTH when we were under a header
        # and when we were already in an unnamed (orphan) section—so split
        # multiple orphan blocks into separate sections.
        if buf and prev_blank and line.strip() and not _HEADER_RE.match(line):
            flush_any()
            title = None  # start a new orphan section

        buf.append(line)
        prev_blank = line.strip() == ""

    # Empty document → one empty section to simplify downstream code
    if not sections and not buf and title is None:
        return [("", "")]

    # Close the final section (empty body allowed if we had a title)
    flush_any()
    return sections


class MarkdownParser(DocumentParser):
    """Markdown parser that produces a Document with raw text.

    Notes:
    - Section splitting is handled by parse_markdown().
    - This parser only fulfills the DocumentParser port:
      can_parse(path)->bool, parse(path)->Document.
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
