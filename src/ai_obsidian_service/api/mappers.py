from __future__ import annotations

from typing import Any, Protocol

from ai_obsidian_service.api.schemas import SearchHitDTO, SearchResponse
from ai_obsidian_service.domain.models import Hit, Query


class ResolveMeta(Protocol):
    def __call__(self, *, chunk_id: str) -> dict[str, Any]: ...

def hit_to_search_hit(hit: Hit, resolve_meta: ResolveMeta) -> SearchHitDTO:
    meta = hit.metadata or (hit.chunk.metadata if hit.chunk else None) or {}
    cid = hit.chunk_id or (hit.chunk.id if hit.chunk else "")
    doc_path = meta.get("path") or meta.get("doc_path")
    return SearchHitDTO(
        id=cid,
        score=hit.score,
        text=(hit.snippet or (hit.chunk.text if hit.chunk else ""))[:2000],
        preview=hit.snippet,
        meta=meta,
        path=meta.get("path"),
        kind=meta.get("kind"),
        doc_path=doc_path,
        chunk_id=cid,
    )

def hits_to_search_response(query: Query, hits: list[Hit], resolve_meta: ResolveMeta) -> SearchResponse:
    return SearchResponse(query=query.text, top_k=query.top_k, hits=[hit_to_search_hit(h, resolve_meta) for h in hits])

# Для обратной совместимости
def answer_to_answer_response(query: str, answer: str, hits: list[Hit], resolve_meta: ResolveMeta):
    return answer, [hit_to_search_hit(h, resolve_meta) for h in hits]
