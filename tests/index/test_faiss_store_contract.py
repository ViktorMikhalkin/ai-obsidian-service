import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # skip entire module if FAISS is not available

from ai_obsidian_service.core import Chunk
from ai_obsidian_service.domain.models import EmbeddedChunk
from ai_obsidian_service.index.faiss_store import FaissVectorStore


def _vec(x: list[float] | np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32)
    # normalize (cosine via IP)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def test_add_and_search_topk():
    store = FaissVectorStore()

    c1 = EmbeddedChunk(chunk=Chunk(id="c1", text="alpha", meta={}), embedding=_vec([1, 0, 0]))
    c2 = EmbeddedChunk(chunk=Chunk(id="c2", text="beta",  meta={}), embedding=_vec([0.9, 0.1, 0]))
    c3 = EmbeddedChunk(chunk=Chunk(id="c3", text="gamma", meta={}), embedding=_vec([0, 1, 0]))

    store.upsert([c1, c2, c3])

    q = _vec([1, 0, 0])
    res = store.search(q, top_k=2)
    assert len(res.hits) == 2
    assert [h.chunk.id for h in res.hits] == ["c1", "c2"]
    assert all(isinstance(h.score, float) for h in res.hits)


def test_update_existing_id_replaces_vector():
    store = FaissVectorStore()
    c1_v1 = EmbeddedChunk(chunk=Chunk(id="c1", text="same", meta={}), embedding=_vec([1, 0, 0]))
    c1_v2 = EmbeddedChunk(chunk=Chunk(id="c1", text="same", meta={}), embedding=_vec([0, 1, 0]))
    c2 = EmbeddedChunk(chunk=Chunk(id="c2", text="other", meta={}), embedding=_vec([0, 0, 1]))

    store.upsert([c1_v1, c2])

    # nearest to x-axis initially -> c1 wins
    res1 = store.search(_vec([1, 0, 0]), top_k=1)
    assert [h.chunk.id for h in res1.hits] == ["c1"]

    # update same id with y-axis vector
    store.upsert([c1_v2])

    # now nearest to y-axis -> c1 still wins but by new vector
    res2 = store.search(_vec([0, 1, 0]), top_k=1)
    assert [h.chunk.id for h in res2.hits] == ["c1"]


def test_empty_store_returns_empty_result():
    store = FaissVectorStore()
    res = store.search(_vec([1, 0, 0]), top_k=3)
    assert res.hits == []


def test_dimension_mismatch_raises():
    store = FaissVectorStore()
    c1 = EmbeddedChunk(chunk=Chunk(id="c1", text="", meta={}), embedding=_vec([1, 0, 0]))
    store.upsert([c1])

    # try to upsert vector with different dim
    c_bad = EmbeddedChunk(chunk=Chunk(id="c2", text="", meta={}), embedding=_vec([1, 0, 0, 0]))
    with pytest.raises(ValueError):
        store.upsert([c_bad])

    # try to search with wrong dim
    with pytest.raises(ValueError):
        store.search(np.asarray([1, 0, 0, 0], dtype=np.float32), top_k=1)
