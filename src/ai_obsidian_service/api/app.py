from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from typing import cast

from fastapi import Depends, FastAPI

from ai_obsidian_service import __version__
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.config.container import build_search_service
from ai_obsidian_service.core import Query

from .mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse


def _as_hits(obj):
    try:
        return list(obj.hits)
    except AttributeError:
        return list(obj)


app = FastAPI(title="AI Obsidian Service", version=__version__)


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    index_dir = os.getenv("AI_OBSIDIAN_INDEX_DIR", None)
    return build_search_service(index_dir=index_dir)


@app.get("/health")
def health():
    return {"ok": True, "errors": []}


@app.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest, svc: SearchService = Depends(get_search_service)
) -> SearchResponse:
    result = svc.search_text(req.query, req.top_k)
    hits = _as_hits(result)
    return hits_to_search_response(
        Query(text=req.query, top_k=req.top_k),
        hits,
        cast(Callable[..., ResolveMeta], svc.resolve_meta),
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(
    req: AnswerRequest, svc: SearchService = Depends(get_search_service)
) -> AnswerResponse:
    result = svc.search_text(req.query, req.top_k)
    hits = _as_hits(result)
    dto_hits = [
        hit_to_search_hit(h, cast(Callable[..., ResolveMeta], svc.resolve_meta))
        for h in hits
    ]
    answer_text = " ".join(h.preview for h in dto_hits if h.preview) or "No answer."
    return AnswerResponse(query=req.query, answer=answer_text, sources=dto_hits)
