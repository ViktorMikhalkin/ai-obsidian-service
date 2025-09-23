import mimetypes
from pathlib import Path

from ai_obsidian_service.core import DocId, Document, DocumentParser


class EpubParser(DocumentParser):
    """EPUB parser (lightweight placeholder)."""
    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".epub"

    def parse(self, path: str) -> Document:
        p = Path(path)
        mime = mimetypes.guess_type(str(p))[0] or "application/epub+zip"
        return Document(id=DocId(str(p)), path=str(p), mime=mime, text="")
