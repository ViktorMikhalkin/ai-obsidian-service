from __future__ import annotations

from typing import Any

from ai_obsidian_service.core import (
    Chunker,
    ChunkId,
    DocId,
    Document,
    DocumentParser,
    EmbeddingIndex,
    Hit,
    Query,
)
from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery


def _embed_text(text: str, dim: int = 64):
    import numpy as np

    vec = np.zeros(dim, dtype=float)
    if text:
        for i, ch in enumerate(text):
            vec[(ord(ch) + i) % dim] += 1.0
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


class SearchService:
    """Orchestrates parsing → chunking → indexing and search over EmbeddingIndex."""

    def get_stats(self) -> dict[str, object]:
        docs = {k[0] for k in self._meta.keys()}
        return {
            "total_documents": len(docs),
            "total_chunks": len(self._meta),
            "errors": [],
        }

    """Orchestrates parsing → chunking → indexing and search over EmbeddingIndex."""

    def __init__(
        self, parsers: list[DocumentParser], chunker: Chunker, index: EmbeddingIndex
    ) -> None:
        self.parsers = list(parsers)
        self.chunker = chunker
        self.index = index
        # local metadata cache for resolve_meta; key = (doc_id, order)
        self._meta: dict[tuple[str, int], dict[str, Any]] = {}

    def index_document(self, doc: Document) -> int:
        chunks = self.chunker.split(doc)
        for ch in chunks:
            key = (str(ch.doc_id), ch.order)
            self._meta[key] = {
                "path": doc.path,
                "kind": doc.mime or "chunk",
                "preview": ch.text[:240] if ch.text else "",
            }
        dim = getattr(self.index, "dim", 64)
        embedded = [
            EmbeddedChunk(chunk=ch, embedding=_embed_text(ch.text, dim))
            for ch in chunks
        ]
        self.index.upsert(embedded)
        return len(chunks)

    def index_path(self, path: str) -> int:
        for p in self.parsers:
            if p.can_parse(path):
                doc = p.parse(path)
                return self.index_document(doc)
        raise ValueError(f"No parser available for: {path}")

    def search_text(self, text: str, top_k: int = 5) -> list[Hit]:
        dim = getattr(self.index, "dim", 64)
        eq = EmbeddedQuery(
            query=Query(text=text, top_k=top_k), embedding=_embed_text(text, dim)
        )
        result = self.index.search(eq)
        return result.hits

    def resolve_meta(
        self, doc_id: DocId, chunk_id: ChunkId, order: int
    ) -> dict[str, Any]:
        key = (str(doc_id), int(order))
        return dict(self._meta.get(key, {}))
