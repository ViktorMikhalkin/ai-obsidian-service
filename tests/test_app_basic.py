from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ai_obsidian_service.domain.models import ChunkId, DocId, Hit, Query, SearchResult
from ai_obsidian_service.indexer.services import IndexerService
from ai_obsidian_service.main import app, get_indexer_service


def _route_exists(path: str, method: str = "GET") -> bool:
    method = method.upper()
    for r in app.router.routes:
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
            return True
    return False


@pytest.fixture
def mock_indexer_service():
    """Create a mock IndexerService for unit testing."""
    mock_service = Mock(spec=IndexerService)

    # Mock search_text method to return a proper SearchResult
    mock_result = SearchResult(
        query=Query(text="test", top_k=1),
        hits=[
            Hit(
                chunk_id=ChunkId("test-chunk-1"),
                doc_id=DocId("test-doc.md"),
                chunk_order=0,
                score=0.95,
                snippet="This is test content for the search query",
                start_char=0,
                end_char=42,
                metadata={"kind": "markdown"},
            )
        ],
        total_time_ms=15.5,
        retrieved_at="2025-01-01T00:00:00Z",
    )
    mock_service.search_text.return_value = mock_result

    # Mock other service methods that might be called
    mock_service.get_stats.return_value = {
        "total_chunks": 100,
        "total_documents": 10,
        "index_size_mb": 5.2,
        "last_updated": "2025-01-01T00:00:00Z",
    }

    return mock_service


def test_openapi_available():
    with TestClient(app) as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert "paths" in data and isinstance(data["paths"], dict)


def test_health_exists_and_ok():
    if not _route_exists("/health", "GET"):
        pytest.skip("No /health endpoint in this app version")
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert "ok" in data


def test_search_minimal_contract(mock_indexer_service):
    """Test search endpoint with mocked service - this is now a proper unit test."""
    if not _route_exists("/search", "POST"):
        pytest.skip("No /search endpoint in this app version")

    # Override the dependency with our mock
    app.dependency_overrides[get_indexer_service] = lambda: mock_indexer_service

    try:
        with TestClient(app) as client:
            r = client.post("/search", json={"query": "test", "top_k": 1})
            assert r.status_code == 200
            body = r.json()

            # Verify the response structure
            assert isinstance(body, dict)
            assert "results" in body
            assert isinstance(body["results"], list)
            assert len(body["results"]) == 1

            # Verify the SearchHit structure
            hit = body["results"][0]
            assert "id" in hit
            assert "path" in hit
            assert "kind" in hit
            assert "preview" in hit
            assert "score" in hit

            # Verify the service was called correctly
            mock_indexer_service.search_text.assert_called_once_with("test", 1)

    finally:
        # Clean up the dependency override
        app.dependency_overrides.clear()


def test_search_response_fields(mock_indexer_service):
    """Test that search response includes all expected fields."""
    if not _route_exists("/search", "POST"):
        pytest.skip("No /search endpoint in this app version")

    app.dependency_overrides[get_indexer_service] = lambda: mock_indexer_service

    try:
        with TestClient(app) as client:
            r = client.post("/search", json={"query": "test query", "top_k": 2})
            assert r.status_code == 200
            body = r.json()

            # Check enhanced response fields
            assert "results" in body
            if "total_time_ms" in body:
                assert isinstance(body["total_time_ms"], (int, float))
            if "retrieved_at" in body:
                assert isinstance(body["retrieved_at"], str)

    finally:
        app.dependency_overrides.clear()


def test_search_error_handling():
    """Test search endpoint error handling when service fails."""
    if not _route_exists("/search", "POST"):
        pytest.skip("No /search endpoint in this app version")

    # Create a mock that raises an exception
    failing_mock = Mock(spec=IndexerService)
    failing_mock.search_text.side_effect = Exception("Service unavailable")

    app.dependency_overrides[get_indexer_service] = lambda: failing_mock

    try:
        with TestClient(app) as client:
            r = client.post("/search", json={"query": "test", "top_k": 1})
            assert r.status_code == 500
            body = r.json()
            assert (
                all(k in body for k in ("code", "message", "requestId"))
                and "detail" not in body
            )
            assert "Service unavailable" in body["message"]

    finally:
        app.dependency_overrides.clear()
