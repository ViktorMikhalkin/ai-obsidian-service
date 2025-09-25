import numpy as np
import pytest

from ai_obsidian_service.domain.models import Chunk, EmbeddedChunk
from ai_obsidian_service.index.faiss_store import FaissVectorStore

pytestmark = pytest.mark.faiss

def _vec(x):
    v = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(v)
    return v/(n+1e-12)

def test_faiss_store_basic():
    store = FaissVectorStore()
    c1 = EmbeddedChunk(chunk=Chunk("c1","d",0,"alpha"), embedding=_vec([1,0,0]))
    c2 = EmbeddedChunk(chunk=Chunk("c2","d",1,"beta"),  embedding=_vec([0.9,0.1,0]))
    store.upsert([c1,c2])
    res = store.search(_vec([1,0,0]), top_k=1)
    assert all(h.chunk is not None for h in res.hits)
    assert [h.chunk.id for h in res.hits if h.chunk is not None] == ["c1"]

