from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ai_obsidian_service.core import Chunk, Chunker, Document, DocumentParser, Query
from ai_obsidian_service.domain.models import SearchResult


class SearchService:
    """
    Single application service for indexing/search for the top level of the application.
    Contract:
      - index_document(doc) -> int
      - index_path(path) -> int
      - search_text(text, top_k) -> SearchResult
      - resolve_meta(result) -> SearchResult
      - shutdown() -> None
    """

    def __init__(
            self,
            *,
            parsers: Iterable[DocumentParser],
            chunker: Chunker,
            index: Any,  # object implementing upsert(chunks) and search(Query) -> SearchResult
    ) -> None:
        self.parsers = list(parsers)
        self.chunker = chunker
        self.index = index

    # ---------- indexing ----------

    def index_document(self, doc: Document) -> int:
        """Split document into chunks and put them into the index. Returns number of chunks."""
        chunks: list[Chunk] = list(self.chunker.split(doc))
        self.index.upsert(chunks)
        return len(chunks)

    def index_path(self, path: str) -> int:
        """Parse file by path and index it."""
        for p in self.parsers:
            if p.can_parse(path):
                doc: Document = p.parse(path)
                return self.index_document(doc)
        return 0

    # ---------- search ----------

    def search_text(self, text: str, top_k: int = 5) -> SearchResult:
        """
        Search by text query — service accepts raw text,
        forms Query and delegates to index.
        """
        q = Query(text=text, top_k=top_k)
        result: SearchResult = self.index.search(q)
        return result

    def resolve_meta(self, result: SearchResult) -> SearchResult:
        """
        Hook for enriching hit metadata (preview, paths, etc.).
        Current implementation is passthrough.
        """
        return result

    # ---------- lifecycle ----------

    def shutdown(self) -> None:
        """Proper shutdown and resource cleanup for index (if required)."""
        close = getattr(self.index, "close", None)
        if callable(close):
            close()
