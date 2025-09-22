from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import cast

from fastapi import Depends, FastAPI, HTTPException

from .api.mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .api.schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse
from .config.container import build_search_service
from .indexer.services import IndexerService  # alias of SearchService

app = FastAPI(title="AI Obsidian Service")


def _as_hits(obj):
    """Accept either List[Hit] or an object with .hits (e.g. SearchResult)."""
    try:
        return list(obj.hits)
    except AttributeError:
        return list(obj)


@lru_cache(maxsize=1)
def get_indexer_service() -> IndexerService:
    index_dir = os.getenv("AI_OBSIDIAN_INDEX_DIR", None)
    svc = build_search_service(index_dir=index_dir)
    return cast(IndexerService, svc)


@app.get("/health")
def health():
    # tests expect exactly this payload
    return {"ok": True, "errors": []}


@app.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest, svc: IndexerService = Depends(get_indexer_service)
) -> SearchResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        from .core import (
            Query,  # local import to avoid potential circular imports at type-check time
        )

        resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
        return hits_to_search_response(
            Query(text=req.query, top_k=req.top_k), hits, resolve
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/answer", response_model=AnswerResponse)
def answer(
    req: AnswerRequest, svc: IndexerService = Depends(get_indexer_service)
) -> AnswerResponse:
    result = svc.search_text(req.query, req.top_k)
    hits = _as_hits(result)
    resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
    dto_hits = [hit_to_search_hit(h, resolve) for h in hits]
    answer_text = " ".join(h.preview for h in dto_hits if h.preview) or "No answer."
    return AnswerResponse(query=req.query, answer=answer_text, sources=dto_hits)
