from __future__ import annotations
from collections.abc import Callable
from typing import TypedDict

from ai_obsidian_service.core import DocId, ChunkId, Hit, Query
from .schemas import SearchHit, SearchResponse, AnswerResponse

class ResolveMeta(TypedDict, total=False):
    path: str
    kind: str
    preview: str
    text: str
    score: float

ResolveFn = Callable[..., ResolveMeta]

def _call_resolve(resolve: ResolveFn, doc_id: DocId, chunk_id: ChunkId, order: int) -> ResolveMeta:
    try:
        return resolve(doc_id, chunk_id, order)
    except TypeError:
        try:
            return resolve(doc_id, order)
        except TypeError:
            return resolve(doc_id)

def hit_to_search_hit(hit: Hit, resolve: ResolveFn) -> SearchHit:
    meta = _call_resolve(resolve, hit.doc_id, hit.chunk_id, hit.chunk_order) or {}
    path = meta.get("path", "")
    kind = meta.get("kind", "chunk")
    preview = meta.get("preview") or meta.get("text", "")
    preview = (preview or "")[:240]
    return SearchHit(
        id=str(hit.chunk_id),  # DTO id must be string
        path=path,
        kind=kind,
        preview=preview,
        score=float(hit.score),
    )

def hits_to_search_response(query: Query, hits: list[Hit], resolve: ResolveFn) -> SearchResponse:
    return SearchResponse(results=[hit_to_search_hit(h, resolve) for h in hits])

def answer_to_answer_response(query_text: str, answer_text: str, sources: list[SearchHit]) -> AnswerResponse:
    return AnswerResponse(query=query_text, answer=answer_text, sources=sources)