
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import DocId, Document
from ai_obsidian_service.domain.models import EmbeddedChunk, Hit, SearchResult
from ai_obsidian_service.index.embedding_index import EmbeddingIndex


class FakeEmbedder:
    dim = 4
    def embed(self, text: str) -> NDArray[np.float32]:
        v = np.zeros(self.dim, dtype=np.float32)
        v[:] = len(text) % 7
        return v

class FakeStore:
    def __init__(self) -> None:
        self.rows: list[EmbeddedChunk] = []
    def upsert(self, chunks: list[EmbeddedChunk]) -> None:
        self.rows.extend(chunks)
    def search(self, query_vec: NDArray[np.float32], top_k: int) -> SearchResult:
        hits = [Hit(chunk=e.chunk, score=float(i)) for i, e in enumerate(self.rows)]
        return SearchResult(query=None, hits=hits[:top_k])  # type: ignore[arg-type]

def test_idempotent_upsert(tmp_path) -> None:
    idx = EmbeddingIndex(embedder=FakeEmbedder(), store=FakeStore(), chunker=SimpleChunker(), state_dir=str(tmp_path))
    doc = Document(id=DocId("D"), path="D.md", mime="text/markdown", text="hello", metadata={"doc_hash": "H1"})
    n1 = idx.index_document(doc)
    n2 = idx.index_document(doc)
    assert n1 >= 0 and n2 == 0

def test_topk_and_order_stability(tmp_path) -> None:
    idx = EmbeddingIndex(embedder=FakeEmbedder(), store=FakeStore(), chunker=SimpleChunker(max_chars=5, overlap=0), state_dir=str(tmp_path))
    doc = Document(id=DocId("D"), path="D.md", mime="text/markdown", text="abcdefghij", metadata={"doc_hash": "H"})
    idx.index_document(doc)
    res = idx.search(text="q", top_k=2)
    assert len(res.hits) == 2
    assert [h.score for h in res.hits] == [0.0, 1.0]
