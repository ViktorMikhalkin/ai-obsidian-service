from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from ai_obsidian_service.core import Chunk, Chunker, Document, DocumentParser
from ai_obsidian_service.domain.models import (
    EmbeddedChunk,
    EmbeddedQuery,
    Query,
    SearchResult,
)


class SearchService:
    """
    Unified app-facing service for indexing and search.

    Public API:
      - index_document(doc) -> int
      - index_path(path) -> int
      - search_text(text, top_k) -> SearchResult
      - resolve_meta(result) -> SearchResult
      - get_stats() -> dict
      - shutdown() -> None
    """

    def __init__(
        self,
        *,
        parsers: Iterable[DocumentParser],
        chunker: Chunker,
        index: Any,  # must expose: upsert(Iterable[EmbeddedChunk]); search_text(str,int) or search(EmbeddedQuery)
    ) -> None:
        self.parsers = list(parsers)
        self.chunker = chunker
        self.index = index

        dim = getattr(self.index, "dim", None)
        if dim is None:
            get_dim = getattr(self.index, "get_dim", None)
            if callable(get_dim):
                try:
                    dim = int(get_dim())
                except Exception:
                    dim = None
        self._dim: int = int(dim or 64)

    # ---------------- internal: tiny deterministic embedder ---------------- #

    def _vec(self, text: str | None) -> np.ndarray:
        """Stable, dependency-light embedding for tests/integration."""
        d = self._dim
        v = np.zeros(d, dtype=float)
        if text:
            for i, ch in enumerate(text):
                v[(ord(ch) + i) % d] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    # ------------------------------- indexing ------------------------------ #

    def index_document(self, doc: Document) -> int:
        """Split a parsed document and upsert embedded chunks into the index."""
        chunks: list[Chunk] = list(self.chunker.split(doc))
        embedded: list[EmbeddedChunk] = [
            EmbeddedChunk(chunk=c, embedding=self._vec(c.text)) for c in chunks
        ]
        self.index.upsert(embedded)
        return len(embedded)

    def index_path(self, path: str) -> int:
        """Parse a path with the first matching parser and index it."""
        for p in self.parsers:
            if p.can_parse(path):
                doc: Document = p.parse(path)
                return self.index_document(doc)
        return 0

    # -------------------------------- search ------------------------------- #

    def search_text(self, text: str, top_k: int = 5) -> SearchResult:
        """
        Perform search for a raw-text query.

        Prefers index.search_text(text, top_k). If unavailable, falls back to
        index.search(EmbeddedQuery(...)).
        """
        if hasattr(self.index, "search_text"):
            return self.index.search_text(text, top_k)  # type: ignore[no-any-return]

        if hasattr(self.index, "search"):
            q = Query(text=text, top_k=top_k)
            eq = EmbeddedQuery(query=q, embedding=self._vec(text))
            return self.index.search(eq)  # type: ignore[no-any-return]

        raise NotImplementedError(
            "Index must implement either search_text(text, top_k) or search(EmbeddedQuery)."
        )

    def resolve_meta(self, result: SearchResult) -> SearchResult:
        """Hook for enriching hits with metadata; passthrough for now."""
        return result

    # --------------------------- observability ----------------------------- #

    def get_stats(self) -> dict[str, int | float | str]:
        """Provide minimal stats; call index.get_stats() if available."""
        get_stats = getattr(self.index, "get_stats", None)
        if callable(get_stats):
            try:
                stats = get_stats()
                if isinstance(stats, dict):
                    return stats
            except Exception:
                pass
        return {
            "documents": 0,
            "chunks": 0,
            "engine": getattr(self.index, "__class__", type("X", (), {})).__name__,
        }

    # ----------------------------- lifecycle ------------------------------- #

    def shutdown(self) -> None:
        close = getattr(self.index, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
