from __future__ import annotations
import re
from pathlib import Path
from typing import List, Tuple, Iterable, Union

__all__ = ["parse_markdown", "MarkdownParser"]

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

def parse_markdown(text: str) -> List[Tuple[str, str]]:
    """Split Markdown into sections by ATX headers.

    Returns a list of (title, body) tuples:
    - title: header text (empty string for preface before first header)
    - body: content under that header, stripped of leading/trailing whitespace
    """
    lines = text.splitlines()
    sections: List[Tuple[str, str]] = []
    title: str | None = None
    body_lines: list[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        m = _HEADER_RE.match(line.strip())
        if m:
            if title is not None or body_lines:
                sections.append((title or "", "\n".join(body_lines).strip()))
                body_lines = []
            title = m.group(2).strip()
        else:
            body_lines.append(line)

    if title is None and not body_lines:
        return [("", "")]  # empty doc edge-case
    sections.append((title or "", "\n".join(body_lines).strip()))
    return sections

class MarkdownParser:
    """Adapter-style parser for Markdown.

    Minimal contract used in tests/services:

    - accepts(path) -> bool

    - parse(text) -> list[(title, body)]

    - parse_file(path) -> list[(title, body)]

    """
    _EXTS = {".md", ".markdown"}

    def accepts(self, path: Union[str, Path]) -> bool:
        try:
            p = Path(path)
            return p.suffix.lower() in self._EXTS
        except Exception:
            return False

    def parse(self, text: str) -> List[Tuple[str, str]]:
        return parse_markdown(text)

    def parse_file(self, path: Union[str, Path], encoding: str = "utf-8") -> List[Tuple[str, str]]:
        with open(path, "r", encoding=encoding) as f:
            return self.parse(f.read())