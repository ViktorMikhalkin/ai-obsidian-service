
from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import DocId, Document
from ai_obsidian_service.di import make_components


def test_collection_filter_works():
    cmp = make_components(chunker=SimpleChunker(max_chars=32, overlap=8))
    doc1 = Document(DocId("notes/a.md"), "notes/a.md", "text/markdown", "alpha hello", {"collection": "notes"})
    doc2 = Document(DocId("docs/b.md"), "docs/b.md", "text/markdown", "beta hello", {"collection": "docs"})
    cmp.search.index_document(doc1)
    cmp.search.index_document(doc2)
    res = cmp.search.search_text("hello", top_k=10, collection="notes")
    assert res.hits
    for h in res.hits:
        meta = (h.metadata or (h.chunk.metadata if h.chunk else None)) or {}
        assert meta.get("collection") == "notes"
