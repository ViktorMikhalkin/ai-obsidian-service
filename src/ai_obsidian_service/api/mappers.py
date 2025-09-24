from __future__ import annotations
from typing import Any, Callable, Dict, Iterable, Protocol

from ai_obsidian_service.api.schemas import SearchHitDTO, SearchResponse
from ai_obsidian_service.core import Query
from ai_obsidian_service.domain.models import Hit


class ResolveMeta(Protocol):
    def __call__(self, chunk_id: str) -> Dict[str, Any]: ...


def hit_to_search_hit(hit: Hit, resolve_meta: ResolveMeta) -> SearchHitDTO:
    meta = resolve_meta(hit.chunk.id)
    return SearchHitDTO(
        id=hit.chunk.id,
        score=float(hit.score),
        text=hit.chunk.text,
        preview=meta.get("preview") or meta.get("summary"),
        meta=meta,
    )


def hits_to_search_response(query: Query, hits: Iterable[Hit], resolve_meta: ResolveMeta) -> SearchResponse:
    dto_hits = [hit_to_search_hit(h, resolve_meta) for h in hits]
    return SearchResponse(query=query.text, top_k=query.top_k, hits=dto_hits)
