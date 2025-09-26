from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Optional

from ai_obsidian_service.domain.models import Document
from ai_obsidian_service.utils.ids import doc_hash, source_id
from ai_obsidian_service.utils.paths import collection_of


class EpubParser:
    """
    EPUB → text parser using ebooklib + BeautifulSoup.
    Extracts HTML chapters and strips tags to plain text.
    """

    exts: tuple[str, ...] = (".epub",)

    def can_parse(self, path: str) -> bool:
        return Path(path).suffix.lower() in self.exts

    def parse(self, vault_root: str, absolute_path: str) -> Optional[Document]:
        try:
            from ebooklib import epub  # lazy import
            from bs4 import BeautifulSoup
        except Exception:
            # deps not installed
            return None

        abs_p = Path(absolute_path)
        if not abs_p.exists():
            return None

        # vault-relative POSIX path
        rel = abs_p.resolve().relative_to(Path(vault_root).resolve())
        rel_posix = PurePosixPath(rel).as_posix()

        try:
            book = epub.read_epub(str(abs_p))
        except Exception:
            return None

        parts: list[str] = []
        for item in book.get_items():
            # ITEM_DOCUMENT is HTML chapter
            if getattr(item, "get_type", None) and item.get_type() == epub.ITEM_DOCUMENT:
                try:
                    html = item.get_content().decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html, "html.parser")
                    txt = soup.get_text(separator=" ", strip=True)
                    if txt:
                        parts.append(txt)
                except Exception:
                    continue

        full_text = "\n\n".join(p for p in parts if p)
        if not full_text.strip():
            return None

        return Document(
            id=source_id(rel_posix),
            text=full_text,
            meta={
                "path": rel_posix,
                "collection": collection_of(rel_posix),
                "doc_hash": doc_hash(full_text),
            },
        )
