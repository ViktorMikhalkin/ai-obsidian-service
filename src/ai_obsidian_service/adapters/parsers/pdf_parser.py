from __future__ import annotations

from pathlib import Path, PurePosixPath

from ai_obsidian_service.domain.models import Document
from ai_obsidian_service.utils.ids import doc_hash, source_id
from ai_obsidian_service.utils.paths import collection_of


class PdfParser:
    """
    Fast PDF → text parser using PyMuPDF (fitz).

    Features:
    - 5-10x faster than pypdf
    - Better text extraction quality
    - Falls back to pypdf if PyMuPDF unavailable
    - Automatic OCR version detection
    - Detects scanned PDFs that need OCR

    OCR Detection:
    - Automatically uses *_ocr.pdf if available
    - Flags PDFs with < 50 chars/page as needing OCR
    - Tracks OCR status in metadata
    """

    exts: tuple[str, ...] = (".pdf",)

    def __init__(self):
        """Detect available PDF library at initialization."""
        self._backend = self._detect_backend()

    def _detect_backend(self) -> str:
        """Check which PDF library is available."""
        try:
            import fitz  # noqa: F401

            return "pymupdf"
        except ImportError:
            try:
                import pypdf  # noqa: F401

                return "pypdf"
            except ImportError:
                return "none"

    def can_parse(self, path: str) -> bool:
        if self._backend == "none":
            return False
        return Path(path).suffix.lower() in self.exts

    def parse(self, vault_root: str, absolute_path: str) -> Document | None:
        if self._backend == "pymupdf":
            return self._parse_pymupdf(vault_root, absolute_path)
        elif self._backend == "pypdf":
            return self._parse_pypdf(vault_root, absolute_path)
        return None

    def _parse_pymupdf(self, vault_root: str, absolute_path: str) -> Document | None:
        """Fast parsing with PyMuPDF with OCR detection."""
        try:
            import fitz
        except ImportError:
            return None

        abs_p = Path(absolute_path)
        if not abs_p.exists():
            return None

        rel = abs_p.resolve().relative_to(Path(vault_root).resolve())
        rel_posix = PurePosixPath(rel).as_posix()

        try:
            # Check for OCR-processed version first
            ocr_path = abs_p.parent / f"{abs_p.stem}_ocr.pdf"
            pdf_to_parse = ocr_path if ocr_path.exists() else abs_p

            doc = fitz.open(str(pdf_to_parse))

            # Extract text from all pages and track content
            parts: list[str] = []
            total_chars = 0
            pages_with_text = 0

            for page in doc:
                # "text" mode is fastest; "blocks" preserves more structure
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    parts.append(page_text.strip())
                    total_chars += len(page_text.strip())
                    pages_with_text += 1

            page_count = len(doc)
            is_ocr_version = pdf_to_parse == ocr_path
            doc.close()

            full_text = "\n\n".join(parts)

            # Detect if PDF needs OCR
            # Heuristic: average < 50 chars per page suggests scanned image
            needs_ocr = False
            if not is_ocr_version and page_count > 0:
                avg_chars_per_page = total_chars / page_count if page_count > 0 else 0
                # Flag as needing OCR if:
                # - Less than 50 chars per page on average, OR
                # - Less than 30% of pages have text
                needs_ocr = (
                    avg_chars_per_page < 50 or (pages_with_text / page_count) < 0.3
                )

        except Exception:
            return None

        if not full_text.strip():
            return None

        return Document(
            id=source_id(rel_posix),
            text=full_text,
            path=rel_posix,
            mime="application/pdf",
            metadata={
                "path": rel_posix,
                "collection": collection_of(rel_posix),
                "doc_hash": doc_hash(full_text),
                "pages": page_count,
                "pages_with_text": pages_with_text,
                "parser": "pymupdf",
                "ocr_processed": is_ocr_version,
                "needs_ocr": needs_ocr,
            },
        )

    def _parse_pypdf(self, vault_root: str, absolute_path: str) -> Document | None:
        """Fallback parsing with pypdf (original implementation)."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return None

        abs_p = Path(absolute_path)
        if not abs_p.exists():
            return None

        rel = abs_p.resolve().relative_to(Path(vault_root).resolve())
        rel_posix = PurePosixPath(rel).as_posix()

        try:
            reader = PdfReader(abs_p)
            parts: list[str] = []
            for page in reader.pages:
                t = page.extract_text() or ""
                if t:
                    parts.append(t.strip())
            full_text = "\n\n".join(p for p in parts if p)
        except Exception:
            return None

        if not full_text.strip():
            return None

        return Document(
            id=source_id(rel_posix),
            text=full_text,
            path=rel_posix,
            mime="application/pdf",
            metadata={
                "path": rel_posix,
                "collection": collection_of(rel_posix),
                "doc_hash": doc_hash(full_text),
                "parser": "pypdf",
                "needs_ocr": False,  # pypdf can't detect this reliably
            },
        )
