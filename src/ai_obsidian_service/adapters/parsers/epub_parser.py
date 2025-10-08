from __future__ import annotations

from pathlib import Path, PurePosixPath

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

    def parse(self, vault_root: str, absolute_path: str) -> Document | None:
        try:
            from bs4 import BeautifulSoup
            from ebooklib import epub  # lazy import
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
            # Check if this is a document item (HTML content)
            # In modern ebooklib, use item.get_type() == ebooklib.ITEM_DOCUMENT
            # or check the media_type for HTML
            try:
                # Method 1: Check media type for HTML content
                media_type = getattr(item, "media_type", "")
                is_html = media_type in ("application/xhtml+xml", "text/html")

                # Method 2: Try get_type() if available
                if not is_html and hasattr(item, "get_type"):
                    # ITEM_DOCUMENT value is 9 in ebooklib
                    is_html = item.get_type() == 9

                if is_html:
                    html = item.get_content().decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html, "lxml")
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
            path=rel_posix,
            text=full_text,
            mime="application/epub+zip",
            metadata={
                "path": rel_posix,
                "collection": collection_of(rel_posix),
                "doc_hash": doc_hash(full_text),
            },
        )
