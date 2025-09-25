from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_obsidian_service.core import Chunker, Document
from ai_obsidian_service.domain.models import (
    Chunk,
    EmbeddedChunk,
    EmbeddedQuery,
    Query,
    SearchResult,
)
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.vector_store import VectorStore


@dataclass(slots=True)
class EmbeddingIndex:
    embedder: Embedder
    store: VectorStore
    chunker: Chunker

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        vecs = [self.embedder.embed(c.text) for c in chunks]
        self.store.upsert([EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vecs, strict=False)])

    def index_document(self, doc: Document) -> int:
        chunks = list(self.chunker.split(doc))
        self.upsert(chunks)
        return len(chunks)

    def search(self, embedded_query: EmbeddedQuery | str, top_k: int = 5) -> SearchResult:
        if isinstance(embedded_query, str):
            q = Query(text=embedded_query, top_k=top_k)
            q_vec = self.embedder.embed(q.text)
        else:
            q_vec = embedded_query.vector
        return self.store.search(q_vec, top_k=top_k)
