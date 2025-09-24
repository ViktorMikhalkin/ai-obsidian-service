from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# if you're using a marker
pytestmark = pytest.mark.requires_faiss if hasattr(pytest, "mark") else []


def test_faiss_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # ENV: enable FAISS + index directory
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    monkeypatch.setenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    index_dir = tmp_path / "index-data"
    monkeypatch.setenv("VECTOR_INDEX_DIR", str(index_dir))

    # Import app after setting ENV (if app is already imported earlier, move this to a separate process)
    from ai_obsidian_service.api.app import app  # noqa

    # 1st run: index and search
    doc = tmp_path / "doc.md"
    doc.write_text("# H\nalpha\nbeta\ngamma\n", encoding="utf-8")

    with TestClient(app) as c:
        r = c.post("/index", json={"path": str(doc)})
        assert r.status_code == 200
        assert r.json()["indexed_chunks"] >= 2

        r = c.post("/search", json={"query": "alpha", "top_k": 1})
        assert r.status_code == 200
        first_hit = r.json()["hits"][0]["id"]

    # 2nd run (new client → lifespan → save in finally of previous client already triggered)
    with TestClient(app) as c2:
        r = c2.post("/search", json={"query": "alpha", "top_k": 1})
        assert r.status_code == 200
        second_hit = r.json()["hits"][0]["id"]

    assert first_hit == second_hit

    # make sure index files exist
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "embeddings.npy").exists()
    assert (index_dir / "chunks.jsonl").exists()
    assert (index_dir / "meta.json").exists()
