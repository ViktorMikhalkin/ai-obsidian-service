
from __future__ import annotations

from dataclasses import dataclass

from ai_obsidian_service.core import Chunk, Chunker, Document
from ai_obsidian_service.domain.models import EmbeddedChunk, Query, SearchResult
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.vector_store import VectorStore


@dataclass(slots=True)
class EmbeddingIndex:
    """Facade: composes Embedder + VectorStore; owns indexing/search orchestration."""
    embedder: Embedder
    store: VectorStore
    chunker: Chunker  # delegates splitting to provided chunker

    def index_document(self, doc: Document) -> int:
        chunks: list[Chunk] = list(self.chunker.split(doc))
        if not chunks:
            return 0
        embedded: list[EmbeddedChunk] = [
            EmbeddedChunk(chunk=c, embedding=self.embedder.embed(c.text)) for c in chunks
        ]
        self.store.upsert(embedded)
        return len(embedded)

    def search(self, text: str, top_k: int = 5) -> SearchResult:
        """Embed the text and delegate to store.search."""
        q = Query(text=text, top_k=top_k)
        q_vec = self.embedder.embed(q.text)
        return self.store.search(q_vec, top_k=top_k)
