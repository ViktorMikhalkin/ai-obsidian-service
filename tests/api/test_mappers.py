# tests/api/test_mappers.py
from __future__ import annotations

from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.domain.models import Chunk, Hit, Query, DocId, ChunkId


def test_hits_to_search_response_maps_fields():
    ch = Chunk(
        id="c1",
        doc_id="d1",
        order=0,
        text="hello",
        metadata={"path": "notes/a.md", "collection": "notes"},
    )
    hit = Hit(
        doc_id=DocId("d1"),
        chunk_id=ChunkId("c1"),
        chunk_order=0,
        score=0.9,
        snippet="hello",
        chunk=ch,
    )

    # Provide resolve_meta that returns a path to ensure the mapper fills it
    res = hits_to_search_response(
        Query("hello", top_k=1),
        [hit],
        lambda chunk_id: {"path": "notes/a.md"},
    )

    assert res.query == "hello"
    assert res.top_k == 1
    assert len(res.hits) == 1
    dto = res.hits[0]
    assert dto.id == "c1"
    assert dto.path == "notes/a.md"
    assert dto.collection == "notes"
    assert dto.score == pytest.approx(0.9, rel=1e-6)