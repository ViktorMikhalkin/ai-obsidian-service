
from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.api.schemas import SearchHitDTO
from ai_obsidian_service.domain.models import Chunk, Hit, Query


def test_hits_to_search_response_maps_fields():
    ch = Chunk(id="c1", doc_id="d1", order=0, text="hello", metadata={"path":"notes/a.md","collection":"notes"})
    hit = Hit(chunk=ch, score=0.9)
    res = hits_to_search_response(Query("hello", top_k=1), [hit], lambda chunk_id: {})
    assert isinstance(res.hits[0], SearchHitDTO)
    assert res.hits[0].id == "c1"
    assert res.hits[0].path == "notes/a.md"
