import json
from pathlib import Path

import numpy as np
import pytest

from ai_obsidian_service.core import Chunk
from ai_obsidian_service.domain.models import EmbeddedChunk
from ai_obsidian_service.index.faiss_store import FaissVectorStore


def _vec(x):
    v = np.asarray(x, dtype=np.float32)
    # L2-normalize to mimic production embedder (cosine via inner-product)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def test_save_and_load_roundtrip(tmp_path: Path):
    store = FaissVectorStore()

    c1 = EmbeddedChunk(Chunk(id="c1", text="alpha", meta={"t": 1}), _vec([1, 0, 0]))
    c2 = EmbeddedChunk(Chunk(id="c2", text="beta",  meta={"t": 2}), _vec([0.9, 0.1, 0]))
    store.upsert([c1, c2])

    store.save(tmp_path, model_name="test-model")

    # files should exist
    for name in ["index.faiss", "embeddings.npy", "chunks.jsonl", "meta.json"]:
        assert (tmp_path / name).exists()

    # meta sanity
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["version"] == 1
    assert meta["dim"] in (3, store.dim)
    assert meta["count"] == 2
    assert meta["ids"] == ["c1", "c2"]
    assert meta["model_name"] == "test-model"

    # load and search
    loaded = FaissVectorStore.load(tmp_path, expected_model_name="test-model")
    res = loaded.search(_vec([1, 0, 0]), top_k=2)
    assert [h.chunk.id for h in res.hits] == ["c1", "c2"]


def test_model_name_mismatch_raises(tmp_path: Path):
    store = FaissVectorStore()
    store.upsert([EmbeddedChunk(Chunk(id="c1", text="", meta={}), _vec([1, 0, 0]))])
    store.save(tmp_path, model_name="A")

    with pytest.raises(ValueError):
        FaissVectorStore.load(tmp_path, expected_model_name="B")


def test_empty_store_persists_and_loads(tmp_path: Path):
    store = FaissVectorStore()
    store.save(tmp_path, model_name="empty-model")

    # files should exist (empty index/embeddings and minimal meta)
    assert (tmp_path / "meta.json").exists()
    assert (tmp_path / "embeddings.npy").exists()
    assert (tmp_path / "index.faiss").exists()
    assert (tmp_path / "chunks.jsonl").exists()

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["count"] == 0
    assert meta["dim"] == 0

    loaded = FaissVectorStore.load(tmp_path, expected_model_name="empty-model")
    res = loaded.search(_vec([1, 0, 0]), top_k=1)
    assert res.hits == []


def test_update_then_save_and_load_preserves_latest_vectors(tmp_path: Path):
    store = FaissVectorStore()
    c1a = EmbeddedChunk(Chunk(id="c1", text="alpha", meta={}), _vec([1, 0, 0]))
    c1b = EmbeddedChunk(Chunk(id="c1", text="alpha", meta={}), _vec([0, 1, 0]))
    c2  = EmbeddedChunk(Chunk(id="c2", text="beta",  meta={}), _vec([0, 0, 1]))

    store.upsert([c1a, c2])
    # initially nearest to x-axis: c1 wins
    r1 = store.search(_vec([1, 0, 0]), top_k=1)
    assert [h.chunk.id for h in r1.hits] == ["c1"]

    # update same id with new vector (y-axis)
    store.upsert([c1b])
    # now nearest to y-axis: c1 wins by new vector
    r2 = store.search(_vec([0, 1, 0]), top_k=1)
    assert [h.chunk.id for h in r2.hits] == ["c1"]

    store.save(tmp_path, model_name="persist-update")
    loaded = FaissVectorStore.load(tmp_path, expected_model_name="persist-update")

    # After load, behavior should be the same as before save
    r3 = loaded.search(_vec([0, 1, 0]), top_k=1)
    assert [h.chunk.id for h in r3.hits] == ["c1"]
