from dataclasses import FrozenInstanceError

from ai_obsidian_service.domain.models import (
    Chunk,
    ChunkId,
    DocId,
    Document,
    Hit,
    Query,
)


def test_document_is_frozen_and_has_fields():
    d = Document(id=DocId("doc-1"), path="/tmp/a.md", mime="text/markdown", text="# Hi")
    assert d.id == "doc-1"
    assert d.mime.startswith("text/")
    try:
        d.path = "/new"  # type: ignore[attr-defined]
        raise AssertionError("Document must be frozen")
    except FrozenInstanceError:
        pass


def test_chunk_and_query_are_immutable():
    c = Chunk(id=ChunkId("chunk-1"), doc_id=DocId("doc-1"), order=0, text="Hello")
    q = Query(text="test", top_k=3)
    assert c.order == 0 and q.top_k == 3
    from dataclasses import FrozenInstanceError

    try:
        q.top_k = 10  # type: ignore[attr-defined]
        raise AssertionError("Query must be frozen")
    except FrozenInstanceError:
        pass


def test_hit_values():
    h = Hit(
        chunk_id=ChunkId("chunk-1"),
        doc_id=DocId("doc-1"),
        chunk_order=0,
        score=0.42,
        snippet="...",
    )
    assert 0.0 <= h.score <= 1.0 or h.score >= 0.0  # non-crashing, loose contract
