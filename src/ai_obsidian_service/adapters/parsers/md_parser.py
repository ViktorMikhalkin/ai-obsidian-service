from __future__ import annotations

from pathlib import Path, PurePosixPath

from ai_obsidian_service.domain.models import Document
from ai_obsidian_service.utils.ids import doc_hash, source_id
from ai_obsidian_service.utils.paths import collection_of


class MarkdownParser:
    """
    Markdown → text parser.

    Unified with PdfParser/EpubParser:
    - can_parse(path: str) -> bool
    - parse(vault_root: str, absolute_path: str) -> Document | None
    Returns Document with meta: {"path","collection","doc_hash"} and id=source_id(rel_posix).
    """

    exts: tuple[str, ...] = (".md", ".markdown")

    def can_parse(self, path: str) -> bool:
        try:
            return Path(path).suffix.lower() in self.exts
        except Exception:
            return False

    def parse(self, vault_root: str, absolute_path: str) -> Document | None:
        abs_p = Path(absolute_path)
        if not abs_p.exists():
            return None

        # vault-relative POSIX path (stable across OS)
        rel = abs_p.resolve().relative_to(Path(vault_root).resolve())
        rel_posix = PurePosixPath(rel).as_posix()

        try:
            text = abs_p.read_text(encoding="utf-8")
        except Exception:
            return None

        # Can add light normalization, but for tests this is sufficient as-is
        if not (text or "").strip():
            return None

        return Document(
            id=source_id(rel_posix),
            path=rel_posix,
            mime="text/markdown",
            text=text,
            metadata={
                "path": rel_posix,
                "collection": collection_of(rel_posix),
                "doc_hash": doc_hash(text),
            },
        )

    # Optional utility for property tests
    def parse_text(
        self, text: str, *, vault_root: str = "", source_rel_path: str = "<memory>"
    ) -> Document:
        rel_posix = PurePosixPath(source_rel_path).as_posix()
        return Document(
            id=source_id(rel_posix),
            path=rel_posix,
            mime="text/markdown",
            text=text or "",
            metadata={
                "path": rel_posix,
                "collection": collection_of(rel_posix),
                "doc_hash": doc_hash(text or ""),
            },
        )
