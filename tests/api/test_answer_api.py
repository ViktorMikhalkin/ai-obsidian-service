from __future__ import annotations
from datetime import datetime
import pytest
from fastapi.testclient import TestClient
from ai_obsidian_service.api import app as api_app
from ai_obsidian_service.core import ChunkId, DocId, Hit, Query, SearchResult

class _DummyChunk:
    def __init__(self, text: str, path: str = "notes/A.md", collection: str = "notes", order: int = 0) -> None:
        self.text = text
        self.meta = {"path": path, "collection": collection}
        self.order = order

def _search_result_with_one_hit() -> SearchResult:
    q = Query(text="test query", top_k=5)
    chunk = _DummyChunk("alpha bravo charlie", path="notes/A.md", collection="notes", order=1)
    hit = Hit(
        doc_id=DocId("doc-1"),
        chunk_id=ChunkId("doc-1:1"),
        chunk_order=1,
        score=0.99,
        snippet="alpha bravo",
        chunk=chunk,  # type: ignore[arg-type]
    )
    return SearchResult(query=q, hits=[hit], total_time_ms=1.2, retrieved_at=datetime.now())

@pytest.fixture()
def client() -> TestClient:
    return TestClient(api_app.app)

def test_answer_mini_fallback_returns_citations(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api_app, "_LLM", None, raising=False)
    monkeypatch.setattr(api_app._service, "search_text", lambda q, top_k=5, collection=None: _search_result_with_one_hit())
    resp = client.post("/answer", json={"query": "what is alpha?", "top_k": 5, "collection": "notes"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "answer" in data and isinstance(data["answer"], str)
    assert "citations" in data and isinstance(data["citations"], list)
    assert len(data["citations"]) >= 1
    c0 = data["citations"][0]
    assert c0["chunk_id"] == "doc-1:1"
    assert c0["doc_path"] == "notes/A.md"

def test_answer_with_ollama_client_used(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLLM:
        def generate(self, prompt: str, system: str | None = None) -> str:
            assert "Context:" in prompt and "Question:" in prompt
            return "LLM answer"
    monkeypatch.setattr(api_app, "_LLM", _FakeLLM(), raising=False)
    monkeypatch.setattr(api_app._service, "search_text", lambda q, top_k=5, collection=None: _search_result_with_one_hit())
    resp = client.post("/answer", json={"query": "what is alpha?", "top_k": 5})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["answer"] == "LLM answer"
    assert len(data["citations"]) >= 1
