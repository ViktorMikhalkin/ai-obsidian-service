import numpy as np
from ai_obsidian_service.core import Document, Chunker, Chunk
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.vector_store import VectorStore
from ai_obsidian_service.domain.models import EmbeddedChunk, SearchResult, SearchHit


class TrivialChunker(Chunker):
    def split(self, doc: Document):
        for i, line in enumerate(doc.text.splitlines()):
            t = line.strip()
            if t:
                yield Chunk(id=f"{doc.id}#{i}", text=t, meta={"line": i})


class DetermEmbedder(Embedder):
    def embed(self, text: str) -> np.ndarray:
        h = abs(hash(text))
        return np.array([(h >> (i * 8)) & 0xFF for i in range(4)], dtype=np.float32)


class MemStore(VectorStore):
    def __init__(self):
        self.vecs: list[np.ndarray] = []
        self.chunks: list[EmbeddedChunk] = []

    def upsert(self, chunks):
        for ec in chunks:
            # replace by id semantics
            for i, ex in enumerate(self.chunks):
                if ex.chunk.id == ec.chunk.id:
                    self.chunks[i] = ec
                    self.vecs[i] = ec.embedding.astype(np.float32)
                    break
            else:
                self.chunks.append(ec)
                self.vecs.append(ec.embedding.astype(np.float32))

    def search(self, query_vec, top_k: int) -> SearchResult:
        if not self.vecs:
            return SearchResult(query=None, hits=[])
        V = np.stack(self.vecs, axis=0)
        q = query_vec.astype(np.float32)
        denom = (np.linalg.norm(V, axis=1) * (np.linalg.norm(q) + 1e-12)) + 1e-12
        sims = (V @ q) / denom
        order = np.argsort(-sims)[:top_k]
        hits = [SearchHit(chunk=self.chunks[i].chunk, score=float(sims[i])) for i in order]
        return SearchResult(query=None, hits=hits)


def test_index_and_search_topk_stability():
    idx = EmbeddingIndex(embedder=DetermEmbedder(), store=MemStore(), chunker=TrivialChunker())
    doc = Document(id="d1", path="d1", mime="text/markdown", text="alpha\nbeta\ngamma\nbeta")
    assert idx.index_document(doc) == 4

    r1 = idx.search("beta", top_k=2)
    r2 = idx.search("beta", top_k=2)

    assert [h.chunk.id for h in r1.hits] == [h.chunk.id for h in r2.hits]
    assert len(r1.hits) == 2


def test_index_empty_document():
    doc = Document(id="empty", path="empty", mime="text/markdown", text="")
    idx = EmbeddingIndex(embedder=DetermEmbedder(), store=MemStore(), chunker=TrivialChunker())
    assert idx.index_document(doc) == 0
    assert idx.search("anything", top_k=3).hits == []
