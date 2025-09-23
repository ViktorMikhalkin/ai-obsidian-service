from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

# Fix: Import the correct dependency function name
from ai_obsidian_service.api.app import app, get_search_service
from ai_obsidian_service.domain.models import ChunkId, DocId, Hit, Query, SearchResult


@pytest.fixture
def mock_indexer_service():
    mock_service = Mock()

    mock_result = SearchResult(
        query=Query(text="test question", top_k=2),  # Changed to top_k=2
        hits=[
            Hit(
                chunk_id=ChunkId("chunk-1"),
                doc_id=DocId("test1.md"),
                chunk_order=0,
                score=0.95,
                snippet="test content 1",
                start_char=0,
                end_char=None,
                metadata=None,
            ),
            Hit(
                chunk_id=ChunkId("chunk-2"),
                doc_id=DocId("test2.md"),
                chunk_order=1,
                score=0.87,
                snippet="test content 2",
                start_char=0,
                end_char=None,
                metadata=None,
            ),
        ],
        total_time_ms=10.5,
        retrieved_at="2025-01-01T00:00:00",
    )
    mock_service.search_text.return_value = mock_result
    mock_service.resolve_meta.return_value = mock_result  # Add resolve_meta mock
    mock_service.llm_client = None

    return mock_service


def test_health_endpoint():
    """Test health endpoint without any dependencies."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True, "errors": []}


def test_search_endpoint_with_mock(mock_indexer_service):
    app.dependency_overrides[get_search_service] = lambda: mock_indexer_service

    try:
        client = TestClient(app)
        response = client.post("/search", json={"query": "test query", "top_k": 2})

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

        # Check that search_text was called
        mock_indexer_service.search_text.assert_called_once_with("test query", 2)

    finally:
        app.dependency_overrides.clear()


def test_answer_endpoint_with_mock(mock_indexer_service):
    """Test answer endpoint with mocked service."""
    app.dependency_overrides[get_search_service] = lambda: mock_indexer_service

    try:
        client = TestClient(app)
        response = client.post(
            "/answer", json={"query": "test question", "top_k": 1}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["query"] == "test question"

    finally:
        app.dependency_overrides.clear()
