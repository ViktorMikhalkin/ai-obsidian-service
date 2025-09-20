from ai_obsidian_service.adapters.index.faiss_index import FaissIndex
from ai_obsidian_service.core import Chunk, DocId, Query


def test_faiss_index_fallback_search():
    idx = FaissIndex(index_dir=None, dim=16)
    chunks = [
        Chunk(doc_id=DocId("d1"), order=0, text="alpha beta gamma"),
        Chunk(doc_id=DocId("d2"), order=0, text="delta epsilon zeta"),
    ]
    idx.upsert(chunks)
    hits = idx.search(Query(text="alpha", top_k=1))
    assert hits and hits[0].doc_id == DocId("d1")
    assert 0.0 <= hits[0].score <= 1.0
