from __future__ import annotations

# Set env BEFORE importing the app, so DI uses memory backend
import os

os.environ.setdefault("VECTOR_STORE_BACKEND", "memory")
os.environ.setdefault("AIOBS_TEST_MODE", "1")
os.environ.setdefault("OLLAMA_BASE_URL", "")
os.environ.setdefault("OLLAMA_MODEL", "")

from datetime import UTC, datetime
from unittest.mock import patch

from fastapi.testclient import TestClient

from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api import app as api_app
from ai_obsidian_service.api.endpoints.config import ServiceConfig
from ai_obsidian_service.domain.models import (
    Chunk,
    ChunkId,
    DocId,
    Hit,
    Query,
    SearchResult,
)


def test_answer_endpoint_minimal(monkeypatch):
    client = TestClient(api_app.app)

    # Patch SearchService.search_text at class level to avoid read-only attribute errors
    def _fake_search(
        self, q: str, top_k: int = 5, collection: str | None = None
    ) -> SearchResult:
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
            retrieved_at=datetime.now(UTC),
        )

    monkeypatch.setattr(SearchService, "search_text", _fake_search, raising=True)

    # Mock config validation to bypass Ollama requirements
    mock_config = ServiceConfig(
        ollama_base_url="http://localhost:11434", ollama_model="test-model"
    )

    with patch(
        "ai_obsidian_service.api.endpoints.search.ConfigValidator"
    ) as MockValidator:
        mock_validator = MockValidator.return_value
        mock_validator.require_rag.return_value = mock_config

        resp = client.post("/answer", json={"query": "hello", "top_k": 1})
        assert resp.status_code == 200, resp.text
        data = resp.json()

        # Accept either 'citations' (older contract) or 'sources' (newer schema)
        citations = None
        if "citations" in data:
            citations = data["citations"]
        elif "sources" in data:
            citations = data["sources"]

        assert isinstance(citations, list) and citations, (
            "no citations/sources returned"
        )
        c0 = citations[0]
        # Minimal keys we rely on
        assert c0.get("chunk_id") == "c1"
        assert c0.get("doc_path") == "notes/a.md"
        span = c0.get("span")
        assert isinstance(span, (list, tuple))
