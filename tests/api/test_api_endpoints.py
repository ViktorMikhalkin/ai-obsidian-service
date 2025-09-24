from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# Make sure this imports your actual app
from ai_obsidian_service.api.app import app


@pytest.fixture(autouse=True)
def env_memory(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # memory backend for speed and predictability in e2e API tests
    monkeypatch.setenv("VECTOR_STORE_BACKEND", "memory")
    monkeypatch.delenv("VECTOR_INDEX_DIR", raising=False)
    yield


def test_health_ok():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json().get("ok") is True


def test_index_and_search_basic(tmp_path):
    # create temporary file for indexing
    p = tmp_path / "doc.md"
    p.write_text("# H\nalpha\nbeta\n", encoding="utf-8")

    with TestClient(app) as c:
        # /index
        r = c.post("/index", json={"path": str(p)})
        assert r.status_code == 200
        assert r.json()["indexed_chunks"] >= 2

        # /search
        r = c.post("/search", json={"query": "alpha", "top_k": 2})
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "alpha"
        assert data["top_k"] == 2
        assert isinstance(data["hits"], list)
        assert all("id" in h and "score" in h and "text" in h for h in data["hits"])

        # /answer
        r = c.post("/answer", json={"query": "alpha", "top_k": 2})
        assert r.status_code == 200
        data = r.json()
        assert data["query"] == "alpha"
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)


def test_index_missing_file_gives_404():
    with TestClient(app) as c:
        r = c.post("/index", json={"path": "/no/such/file.md"})
        assert r.status_code == 404
        j = r.json()
        # unified error: detail is present, and middleware also adds {code,message,requestId}
        assert "detail" in j
