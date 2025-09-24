from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api.app import app, get_search_service
from ai_obsidian_service.domain.models import ChunkId, DocId, Hit, Query, SearchResult


def _route_exists(path: str, method: str = "GET") -> bool:
    method = method.upper()
    for r in app.router.routes:
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set()):
            return True
    return False


@pytest.fixture
def mock_search_service():
    """Create a mock SearchService for unit testing."""
    mock_service = Mock(spec=SearchService)

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
    mock_service.resolve_meta.return_value = mock_result  # Add resolve_meta mock

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
        assert data["ok"] is True


def test_search_minimal_contract(mock_search_service):
    """Test search endpoint with mocked service."""
    if not _route_exists("/search", "POST"):
        pytest.skip("No /search endpoint in this app version")

    # Override the dependency with our mock
    app.dependency_overrides[get_search_service] = lambda: mock_search_service

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

            # Verify the Hit structure
            hit = body["results"][0]
            assert "id" in hit
            assert "path" in hit
            assert "kind" in hit
            assert "preview" in hit
            assert "score" in hit

            # Verify the service was called correctly
            mock_search_service.search_text.assert_called_once_with("test", 1)

    finally:
        # Clean up the dependency override
        app.dependency_overrides.clear()


def test_answer_minimal_contract(mock_search_service):
    """Test answer endpoint with mocked service."""
    if not _route_exists("/answer", "POST"):
        pytest.skip("No /answer endpoint in this app version")

    app.dependency_overrides[get_search_service] = lambda: mock_search_service

    try:
        with TestClient(app) as client:
            r = client.post("/answer", json={"query": "test question", "top_k": 1})
            assert r.status_code == 200
            body = r.json()

            # Verify the response structure
            assert isinstance(body, dict)
            assert "query" in body
            assert "answer" in body
            assert "sources" in body
            assert body["query"] == "test question"
            assert isinstance(body["answer"], str)
            assert isinstance(body["sources"], list)

            # Verify the service was called correctly
            mock_search_service.search_text.assert_called_once_with("test question", 1)

    finally:
        app.dependency_overrides.clear()


def test_search_error_handling():
    """Test search endpoint error handling when service fails."""
    if not _route_exists("/search", "POST"):
        pytest.skip("No /search endpoint in this app version")

    # Create a mock that raises an exception
    failing_mock = Mock(spec=SearchService)
    failing_mock.search_text.side_effect = Exception("Service unavailable")

    app.dependency_overrides[get_search_service] = lambda: failing_mock

    try:
        with TestClient(app) as client:
            r = client.post("/search", json={"query": "test", "top_k": 1})
            assert r.status_code == 500
            body = r.json()
            # Check that we get proper error response structure from error handlers
            assert "code" in body
            assert "message" in body
            assert "requestId" in body
            assert "Service unavailable" in body["message"]

    finally:
        app.dependency_overrides.clear()


def test_service_integration():
    """Test that we can build and use the search service directly."""
    from ai_obsidian_service.config.container import build_search_service
    from ai_obsidian_service.core import DocId, Document

    # Build the service
    service = build_search_service(index_dir=None)

    try:
        # Test basic functionality
        doc = Document(
            id=DocId("test-doc"),
            path="/tmp/test.md",
            mime="text/markdown",
            text="This is a test document with some content."
        )

        # Index a document
        chunks_added = service.index_document(doc)
        assert chunks_added > 0

        # Search for content
        result = service.search_text("test", top_k=1)
        assert result is not None
        assert hasattr(result, 'hits')

        # Test resolve_meta
        resolved = service.resolve_meta(result)
        assert resolved is not None

    finally:
        # Clean up
        service.shutdown()
