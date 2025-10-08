# tests/api/test_mappers.py
from __future__ import annotations

import pytest

from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.domain.models import Chunk, ChunkId, DocId, Hit, Query


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

    # resolve_meta returns path — mapper will fill it into the DTO
    res = hits_to_search_response(
        Query("hello", top_k=1),
        [hit],
        lambda chunk_id: {"path": "notes/a.md"},
    )

    assert res.query == "hello"
    assert res.top_k == 1
    assert len(res.hits) == 1

    h = res.hits[0]
    payload = (
        h.model_dump()
        if hasattr(h, "model_dump")
        else (h if isinstance(h, dict) else h.__dict__)
    )
    assert payload.get("id") == "c1"
    assert payload.get("path") == "notes/a.md"
    assert pytest.approx(payload.get("score", 0.0), rel=1e-6) == 0.9
