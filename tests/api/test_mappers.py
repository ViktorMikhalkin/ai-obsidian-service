from ai_obsidian_service.api.mappers import (
    answer_to_answer_response,
    hit_to_search_hit,
    hits_to_search_response,
)
from ai_obsidian_service.api.schemas import SearchHitDTO as SearchHit
from ai_obsidian_service.domain.models import ChunkId, DocId, Hit, Query


def _resolve3(doc_id: DocId, chunk_id: ChunkId, order: int):
    return {
        "path": f"/docs/{doc_id}.md",
        "kind": "md",
        "preview": f"chunk-{order}",
        "score": 0.42,
    }


def _resolve2(doc_id: DocId, order: int):
    # legacy signature support
    return {
        "path": f"/legacy/{doc_id}.md",
        "kind": "md",
        "preview": f"legacy-{order}",
    }


def test_hit_to_search_hit_basic():
    h = Hit(
        doc_id=DocId("d1"),
        chunk_id=ChunkId("d1#2"),
        chunk_order=2,
        score=0.8,
        snippet="...",
    )
    dto = hit_to_search_hit(h, _resolve3)
    assert dto.path.endswith("/docs/d1.md")
    assert dto.kind == "md"
    assert dto.score == 0.8
    assert "chunk-2" in dto.preview


def test_hit_to_search_hit_legacy_resolver():
    h = Hit(
        doc_id=DocId("d2"),
        chunk_id=ChunkId("d2#0"),
        chunk_order=0,
        score=0.5,
        snippet="s",
    )
    dto = hit_to_search_hit(h, _resolve2)
    assert dto.path.endswith("/legacy/d2.md")
    assert "legacy-0" in dto.preview


def test_hits_to_search_response_list():
    q = Query(text="hello", top_k=2)
    hits = [
        Hit(
            doc_id=DocId("d1"),
            chunk_id=ChunkId("d1#0"),
            chunk_order=0,
            score=0.5,
            snippet="s",
        )
    ]
    resp = hits_to_search_response(q, hits, _resolve3)
    assert len(resp.results) == 1
    assert resp.results[0].path.endswith("/docs/d1.md")


def test_answer_to_answer_response():
    sources: list[SearchHit] = []
    resp = answer_to_answer_response("q", "a", sources)
    assert resp.query == "q"
    assert resp.answer == "a"
    assert isinstance(resp.sources, list)
