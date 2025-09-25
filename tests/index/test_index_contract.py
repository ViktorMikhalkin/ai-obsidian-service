
from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import DocId, Document
from ai_obsidian_service.di import make_components


def test_empty_index_search_returns_empty():
    cmp = make_components(chunker=SimpleChunker(max_chars=32, overlap=8))
    res = cmp.search.search_text("hello", top_k=5)
    assert res.hits == []

def test_search_respects_topk_and_scores():
    cmp = make_components(chunker=SimpleChunker(max_chars=16, overlap=4))
    cmp.search.index_document(Document(DocId("a.md"), "a.md", "text/markdown", "hello world hello"))
    cmp.search.index_document(Document(DocId("b.md"), "b.md", "text/markdown", "another hello line"))
    res = cmp.search.search_text("hello", top_k=3)
    assert 0 < len(res.hits) <= 3
    assert all(getattr(h, "score", None) is not None for h in res.hits)
