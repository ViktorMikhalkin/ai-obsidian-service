import mimetypes
from pathlib import Path

from ai_obsidian_service.core import DocId, Document, DocumentParser


class MarkdownParser(DocumentParser):
    """Markdown file parser (strategy). Lightweight and side-effect free."""

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() in {".md", ".markdown"}

    def parse(self, path: str) -> Document:
        p = Path(path)
        text = p.read_text(encoding="utf-8", errors="ignore")
        mime = mimetypes.guess_type(str(p))[0] or "text/markdown"
        return Document(id=DocId(str(p)), path=str(p), mime=mime, text=text)
