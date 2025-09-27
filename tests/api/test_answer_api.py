# tests/api/test_answer_api.py
from __future__ import annotations

# Set env BEFORE importing the app, so DI uses memory backend
import os
os.environ.setdefault("VECTOR_STORE_BACKEND", "memory")
os.environ.setdefault("AIOBS_TEST_MODE", "1")
os.environ.setdefault("OLLAMA_BASE_URL", "")
os.environ.setdefault("OLLAMA_MODEL", "")

from fastapi.testclient import TestClient
from ai_obsidian_service.api import app as api_app


def test_answer_endpoint_minimal(monkeypatch):
    client = TestClient(api_app.app)

    # Patch SearchService.search_text at class level to avoid read-only attribute errors
    from ai_obsidian_service.adapters.services.search_service import SearchService
    from ai_obsidian_service.domain.models import (
        DocId, ChunkId, Chunk, Hit, Query, SearchResult
    )
    from datetime import datetime

    def _fake_search(self, q: str, top_k: int = 5, collection: str | None = None) -> SearchResult:
        ch = Chunk(
            id="c1",
            doc_id="d1",
            order=0,
            text="hello world",
            metadata={"path": "notes/a.md", "collection": "notes"},
        )
        hit = Hit(
            doc_id=DocId("d1"),
            chunk_id=ChunkId("c1"),
            chunk_order=0,
            score=0.42,
            snippet="hello",
            chunk=ch,
        )
        return SearchResult(
            query=Query(q, top_k=top_k),
            hits=[hit],
            total_time_ms=1.23,
            retrieved_at=datetime.utcnow(),
        )

    monkeypatch.setattr(SearchService, "search_text", _fake_search, raising=True)

    resp = client.post("/answer", json={"query": "hello", "top_k": 1})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Contract the tests expect: citations array with minimal fields
    assert "citations" in data
    assert isinstance(data["citations"], list)
    assert data["citations"], "citations should not be empty"
    c = data["citations"][0]
    assert c["chunk_id"] == "c1"
    assert c["doc_path"] == "notes/a.md"
    assert isinstance(c["span"], list) or isinstance(c["span"], tuple)