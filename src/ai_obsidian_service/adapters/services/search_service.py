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


class SearchService:
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
        self.index.upsert(chunks)
        return len(chunks)

    def index_path(self, path: str) -> int:
        for p in self.parsers:
            if p.can_parse(path):
                doc = p.parse(path)
                return self.index_document(doc)
        raise ValueError(f"No parser available for: {path}")

    def search_text(self, text: str, top_k: int = 5) -> list[Hit]:
        return self.index.search(Query(text=text, top_k=top_k))

    def resolve_meta(
        self, doc_id: DocId, chunk_id: ChunkId, order: int
    ) -> dict[str, Any]:
        key = (str(doc_id), int(order))
        return dict(self._meta.get(key, {}))
