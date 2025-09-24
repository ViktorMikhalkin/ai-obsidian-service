import numpy as np
import pytest

from ai_obsidian_service.core import Document, Chunker, Chunk
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.vector_store import VectorStore
from ai_obsidian_service.domain.models import EmbeddedChunk, SearchResult


class TrivialChunker(Chunker):
    def split(self, doc: Document):
        # one chunk per line, including empty lines filtered out
        for i, line in enumerate(doc.text.splitlines()):
            t = line.strip()
            if t:
                yield Chunk(id=f"{doc.id}#{i}", text=t, meta={"line": i})


class DetermEmbedder(Embedder):
    def embed(self, text: str) -> np.ndarray:
        # deterministic 3D
        h = abs(hash(text))
        return np.array([(h >> (i * 10)) & 0x3FF for i in range(3)], dtype=float)


class MemStore(VectorStore):
    def __init__(self):
        self.vecs = []
        self.chunks = []

    def upsert(self, chunks):
        for ec in chunks:
            self.vecs.append(ec.embedding)
            self.chunks.append(ec)

    def search(self, query_vec, top_k: int) -> SearchResult:
        import numpy as np
        from ai_obsidian_service.domain.models import SearchHit, SearchResult

        if not self.vecs:
            return SearchResult(query=None, hits=[])

        V = np.stack(self.vecs, axis=0)
        q = query_vec.astype(float)
        denom = (np.linalg.norm(V, axis=1) * (np.linalg.norm(q) + 1e-12)) + 1e-12
        sims = (V @ q) / denom
        order = np.argsort(-sims)
        top = order[:top_k].tolist()

        hits = [SearchHit(chunk=self.chunks[i].chunk, score=float(sims[i])) for i in top]
        return SearchResult(query=None, hits=hits)


def test_indexing_and_search_topk_stability():
    doc = Document(id="doc1", path="doc1", mime="text/markdown", text="alpha\nbeta\ngamma\nbeta")
    idx = EmbeddingIndex(embedder=DetermEmbedder(), store=MemStore(), chunker=TrivialChunker())

    n = idx.index_document(doc)
    assert n == 4

    # same query yields deterministic ranking (top_k=2)
    r1 = idx.search("beta", top_k=2)
    r2 = idx.search("beta", top_k=2)

    assert len(r1.hits) == 2
    assert [h.chunk.text for h in r1.hits] == [h.chunk.text for h in r2.hits]
    assert all(isinstance(h.score, float) for h in r1.hits)


def test_index_empty_document():
    doc = Document(id="empty", path="empty", mime="text/markdown", text="")
    idx = EmbeddingIndex(embedder=DetermEmbedder(), store=MemStore(), chunker=TrivialChunker())
    assert idx.index_document(doc) == 0
    assert idx.search("anything", top_k=3).hits == []
