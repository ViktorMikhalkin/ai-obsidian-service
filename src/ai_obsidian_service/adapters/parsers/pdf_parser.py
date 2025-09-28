from __future__ import annotations

from pathlib import Path, PurePosixPath

from ai_obsidian_service.domain.models import Document
from ai_obsidian_service.utils.ids import doc_hash, source_id
from ai_obsidian_service.utils.paths import collection_of


class PdfParser:
    """
    Lightweight PDF → text parser using pypdf.
    Extracts text page-by-page (no OCR).
    """

    exts: tuple[str, ...] = (".pdf",)

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.exts

    def parse(self, vault_root: str, absolute_path: str) -> Document | None:
        try:
            from pypdf import PdfReader  # lazy import
        except Exception:
            # pypdf not installed
            return None

        abs_p = Path(absolute_path)
        if not abs_p.exists():
            return None

        # vault-relative POSIX path
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
            },
        )
