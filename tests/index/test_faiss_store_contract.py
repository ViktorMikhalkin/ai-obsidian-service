import numpy as np
import pytest

from ai_obsidian_service.core import Chunk, DocId
from ai_obsidian_service.domain.models import EmbeddedChunk
from ai_obsidian_service.index.faiss_store import FaissVectorStore

try:
    pass  # type: ignore
except Exception:  # pragma: no cover
    pytest.skip("faiss is not installed", allow_module_level=True)

def _vec(x: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

def _chunk(cid: str, text: str, order: int = 0) -> Chunk:
    return Chunk(id=cid, doc_id=DocId("doc"), order=order, text=text)

def test_add_and_search_topk():
    store = FaissVectorStore()

    c1 = EmbeddedChunk(chunk=_chunk("c1", "alpha", 0), embedding=_vec([1, 0, 0]))
    c2 = EmbeddedChunk(chunk=_chunk("c2", "beta", 1), embedding=_vec([0.9, 0.1, 0]))
    c3 = EmbeddedChunk(chunk=_chunk("c3", "gamma", 2), embedding=_vec([0, 1, 0]))

    store.upsert([c1, c2, c3])

    q = _vec([1, 0, 0])
    res = store.search(q, top_k=2)
    assert len(res.hits) == 2
    assert [h.chunk.id for h in res.hits] == ["c1", "c2"]
    assert all(isinstance(h.score, float) for h in res.hits)

def test_update_existing_id_replaces_vector():
    store = FaissVectorStore()
    c1_v1 = EmbeddedChunk(chunk=_chunk("c1", "same", 0), embedding=_vec([1, 0, 0]))
    c1_v2 = EmbeddedChunk(chunk=_chunk("c1", "same", 0), embedding=_vec([0, 1, 0]))
    c2 = EmbeddedChunk(chunk=_chunk("c2", "other", 1), embedding=_vec([0, 0, 1]))

    store.upsert([c1_v1, c2])
    res1 = store.search(_vec([1, 0, 0]), top_k=1)
    assert [h.chunk.id for h in res1.hits] == ["c1"]

    store.upsert([c1_v2])
    res2 = store.search(_vec([0, 1, 0]), top_k=1)
    assert [h.chunk.id for h in res2.hits] == ["c1"]

def test_empty_store_returns_empty_result():
    store = FaissVectorStore()
    res = store.search(_vec([1, 0, 0]), top_k=3)
    assert res.hits == []

def test_dimension_mismatch_raises():
    store = FaissVectorStore()
    c1 = EmbeddedChunk(chunk=_chunk("c1", "", 0), embedding=_vec([1, 0, 0]))
    store.upsert([c1])

    c_bad = EmbeddedChunk(chunk=_chunk("c2", "", 1), embedding=_vec([1, 0, 0, 0]))
    with pytest.raises(ValueError):
        store.upsert([c_bad])

    with pytest.raises(ValueError):
        store.search(np.asarray([1, 0, 0, 0], dtype=np.float32), top_k=1)
