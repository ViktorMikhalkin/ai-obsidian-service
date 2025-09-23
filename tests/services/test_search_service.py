from ai_obsidian_service.config.container import build_search_service
from ai_obsidian_service.core import DocId, Document


def test_search_service_index_and_search():
    svc = build_search_service(index_dir=None)
    doc = Document(
        id=DocId("doc1"),
        path="/tmp/doc1.md",
        mime="text/markdown",
        text="hello world " * 20,
    )
    added = svc.index_document(doc)
    assert added > 0

    # Fix: search_text returns SearchResult, not a list
    search_result = svc.search_text("hello", top_k=3)

    # Fix: Access hits through the SearchResult object
    assert len(search_result.hits) >= 1

    # Fix: resolve_meta takes only the SearchResult object, not individual parameters
    resolved_result = svc.resolve_meta(search_result)

    # Fix: Check the resolved result structure
    # Since resolve_meta is currently a passthrough, we check the original search result
    assert len(resolved_result.hits) >= 1
    first_hit = resolved_result.hits[0]

    # Verify the hit has the expected attributes
    assert hasattr(first_hit, 'doc_id')
    assert hasattr(first_hit, 'chunk_id')
    assert hasattr(first_hit, 'snippet')


def test_search_service_lifecycle():
    """Test that the service can be properly shut down."""
    svc = build_search_service(index_dir=None)

    # Should not raise an exception
    svc.shutdown()


def test_search_service_empty_query():
    """Test search with empty or minimal query."""
    svc = build_search_service(index_dir=None)

    # Search without indexing anything
    search_result = svc.search_text("nonexistent", top_k=1)

    # Should return empty results, not error
    assert len(search_result.hits) == 0
