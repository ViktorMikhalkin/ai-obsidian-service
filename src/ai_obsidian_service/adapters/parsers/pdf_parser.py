import mimetypes
from pathlib import Path

from ai_obsidian_service.core import DocId, Document, DocumentParser


class PdfParser(DocumentParser):
    """PDF parser strategy.
    NOTE: Lightweight fallback that does not depend on external PDF libs.
    It returns empty text for now; a real extractor can be plugged later.
    """

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".pdf"

    def parse(self, path: str) -> Document:
        p = Path(path)
        # Fallback: no real text extraction to keep it dependency-light.
        text = ""
        mime = mimetypes.guess_type(str(p))[0] or "application/pdf"
        return Document(id=DocId(str(p)), path=str(p), mime=mime, text=text)
