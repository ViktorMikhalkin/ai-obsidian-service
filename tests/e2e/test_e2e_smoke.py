from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from ai_obsidian_service.api.app import app

# Mark the whole module as e2e so it's easy to select in CI:
pytestmark = [pytest.mark.e2e]


# ---------- Fast e2e with MEMORY (CI-friendly) ----------
@pytest.mark.unit
@pytest.fixture(autouse=True)
def _env_memory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """
    For this module, default to memory backend so the quick e2e runs on any CI.
    Integration tests below override env explicitly in their bodies.
    """
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "memory")
    monkeypatch.delenv("VECTOR_INDEX_DIR", raising=False)
    yield


def _index_files(client: TestClient, paths: list[str]) -> None:
    for p in paths:
        r = client.post("/index", json={"path": p})
        assert r.status_code == 200, r.text
        assert r.json()["indexed_chunks"] >= 1


def test_e2e_memory_smoke(mini_corpus):
    required, optional = mini_corpus
    with TestClient(app) as c:
        # index only the required markdowns (always available)
        _index_files(c, [str(p) for p in required])

        # basic search
        r = c.post("/search", json={"query": "FAISS", "top_k": 5})
        assert r.status_code == 200
        js = r.json()
        assert js["query"] == "FAISS"
        assert isinstance(js["hits"], list)

        # answer
        r = c.post("/answer", json={"query": "How to persist index?", "top_k": 5})
        assert r.status_code == 200
        js = r.json()
        assert "answer" in js
        assert isinstance(js["sources"], list)


# ---------- E2E with FAISS + persist (CPU integration) ----------

@pytest.mark.integration_cpu
@pytest.mark.requires_faiss
@pytest.mark.requires_st
def test_e2e_faiss_persist(mini_corpus, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """
    Integration on real faiss-cpu + sentence-transformers.
    Requires running the suite with USE_FAKE_FAISS=0 and USE_FAKE_ST=0 (see conftest.py notes).
    """
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    idx_dir = tmp_path / "faiss-index"
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(idx_dir))

    required, optional = mini_corpus

    # First run — index and search
    with TestClient(app) as c1:
        _index_files(c1, [str(p) for p in required + optional])

        r = c1.post("/search", json={"query": "Usage", "top_k": 3})
        assert r.status_code == 200
        first_ids = [h["id"] for h in r.json()["hits"]]

    # Second run — new client triggers shutdown(save)/startup(load)
    with TestClient(app) as c2:
        r = c2.post("/search", json={"query": "Usage", "top_k": 3})
        assert r.status_code == 200
        second_ids = [h["id"] for h in r.json()["hits"]]

    assert first_ids[:1] == second_ids[:1]  # top-1 stable across restart

    # persistence files exist
    for name in ["index.faiss", "embeddings.npy", "chunks.jsonl", "meta.json"]:
        assert (idx_dir / name).exists(), f"missing {name}"


# ---------- E2E with FAISS + persist (GPU integration, optional) ----------

@pytest.mark.integration_gpu
@pytest.mark.requires_faiss
@pytest.mark.requires_st
def test_e2e_faiss_persist_gpu(mini_corpus, tmp_path, monkeypatch: pytest.MonkeyPatch):
    """
    Optional GPU variant. Will be auto-skipped by conftest.py if no CUDA/torch/faiss-gpu.
    Run locally with a GPU env and real deps (USE_FAKE_* = 0).
    """
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    idx_dir = tmp_path / "faiss-index-gpu"
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(idx_dir))

    required, optional = mini_corpus

    with TestClient(app) as c1:
        _index_files(c1, [str(p) for p in required + optional])

        r = c1.post("/search", json={"query": "Usage", "top_k": 3})
        assert r.status_code == 200
        first_ids = [h["id"] for h in r.json()["hits"]]

    with TestClient(app) as c2:
        r = c2.post("/search", json={"query": "Usage", "top_k": 3})
        assert r.status_code == 200
        second_ids = [h["id"] for h in r.json()["hits"]]

    assert first_ids[:1] == second_ids[:1]

    for name in ["index.faiss", "embeddings.npy", "chunks.jsonl", "meta.json"]:
        assert (idx_dir / name).exists(), f"missing {name}"
