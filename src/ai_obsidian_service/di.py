from __future__ import annotations

from dataclasses import dataclass

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Chunker, DocumentParser
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.vector_store import VectorStore

# ---- Example concrete implementations (replace with your real ones) ----

class DummyEmbedder(Embedder):
    """Deterministic toy embedder for tests/prototyping."""
    def embed(self, text: str):
        import numpy as np
        # fixed-size 4D vector from hash (deterministic)
        h = abs(hash(text))
        return np.array([(h >> (i * 8)) & 0xFF for i in range(4)], dtype=float)


class InMemoryVectorStore(VectorStore):
    """Simple cosine-sim memory store for prototyping and contract tests."""
    def __init__(self):
        import numpy as np
        self._vecs: list[np.ndarray] = []
        self._chunks = []

    def upsert(self, chunks):
        for ec in chunks:
            self._vecs.append(ec.embedding)
            self._chunks.append(ec)

    def search(self, query_vec, top_k: int):
        import numpy as np

        from ai_obsidian_service.domain.models import SearchHit, SearchResult

        if not self._vecs:
            return SearchResult(query=None, hits=[])

        V = np.stack(self._vecs)  # (N, D)
        q = query_vec.astype(float)
        # cosine sim
        denom = (np.linalg.norm(V, axis=1) * (np.linalg.norm(q) + 1e-12)) + 1e-12
        sims = (V @ q) / denom
        idx = np.argsort(-sims)[:top_k].tolist()

        hits = [SearchHit(chunk=self._chunks[i].chunk, score=float(sims[i])) for i in idx]
        return SearchResult(query=None, hits=hits)


# ---- Factories ----

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
