
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Chunker, DocumentParser
from ai_obsidian_service.domain.models import EmbeddedChunk, Hit, SearchResult
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.vector_store import VectorStore

# ---- Example concrete implementations (replace with your real ones) ----

class DummyEmbedder(Embedder):
    """Deterministic toy embedder for tests/prototyping."""
    def embed(self, text: str) -> np.ndarray:
        # encode length modulo into a tiny vector for determinism
        v = np.zeros(4, dtype=np.float32)
        v[:] = (len(text) % 7)
        return v

class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self.rows: list[EmbeddedChunk] = []

    def upsert(self, chunks: list[EmbeddedChunk]) -> None:  # type: ignore[override]
        self.rows.extend(chunks)

    def count(self) -> int:
        return len(self.rows)

    def search(self, query_vec: np.ndarray, top_k: int) -> SearchResult:  # type: ignore[override]
        # naive scoring by vector[0] closeness to len%7 of chunk text
        qv = float(query_vec[0])
        scored: list[tuple[float, EmbeddedChunk]] = []
        for e in self.rows:
            s = -abs(len(e.chunk.text) % 7 - qv)
            scored.append((s, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        hits = [Hit(chunk=e.chunk, score=float(s)) for s, e in scored[:top_k]]
        return SearchResult(query=None, hits=hits)

@dataclass(slots=True)
class Components:
    embedder: Embedder
    store: VectorStore
    index: EmbeddingIndex
    parser: DocumentParser
    search: SearchService

def make_components(*, chunker: Chunker) -> Components:
    """Wire concrete implementations. Swap here for FAISS/OpenAI/etc."""
    embedder = DummyEmbedder()
    store = InMemoryVectorStore()
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
    search = SearchService(index=index, parser=parser)
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)
