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
    hits = svc.search_text("hello", top_k=3)
    assert len(hits) >= 1
    meta = svc.resolve_meta(hits[0].doc_id, hits[0].chunk_id, hits[0].chunk_order)
    assert "path" in meta and "preview" in meta
