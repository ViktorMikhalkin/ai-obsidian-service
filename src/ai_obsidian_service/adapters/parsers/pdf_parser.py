from pathlib import Path
import mimetypes
from ai_obsidian_service.core import DocumentParser, Document, DocId

class PdfParser(DocumentParser):
    """PDF parser (lightweight placeholder without external deps)."""
    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() == ".pdf"

    def parse(self, path: str) -> Document:
        p = Path(path)
        mime = mimetypes.guess_type(str(p))[0] or "application/pdf"
        # TODO: plug real extractor in future iteration
        return Document(id=DocId(str(p)), path=str(p), mime=mime, text="")
