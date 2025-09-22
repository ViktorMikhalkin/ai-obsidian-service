from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import cast

from fastapi import Depends, FastAPI, HTTPException, Request

from ai_obsidian_service import __version__
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.config.container import build_search_service
from ai_obsidian_service.core import Query

from .errors import (
    ensure_request_id_middleware,
    install_access_logger,
    install_error_handlers,
)
from .mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse


@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    index_dir = os.getenv("AI_OBSIDIAN_INDEX_DIR", None)
    return build_search_service(index_dir=index_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = get_search_service()
    try:
        yield
    finally:
        try:
            svc.shutdown()
        except Exception:
            pass


app = FastAPI(title="AI Obsidian Service", version=__version__, lifespan=lifespan)
ensure_request_id_middleware(app)
install_access_logger(app)
install_error_handlers(app)


def _as_hits(obj):
    try:
        return list(obj.hits)  # SearchResult-like
    except AttributeError:
        return list(obj)  # already list[Hit]


@app.get("/health")
def health(request: Request):
    rid = request.headers.get("X-Request-ID") or getattr(
        request.state, "request_id", "-"
    )
    logging.getLogger("ai_obsidian_service.api.access").info(
        "access",
        extra={
            "requestId": rid,
            "method": "GET",
            "path": "/health",
            "status": 200,
            "duration_ms": 0.0,
        },
    )
    return {"ok": True, "errors": []}


@app.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest, svc: SearchService = Depends(get_search_service)
) -> SearchResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        return hits_to_search_response(
            Query(text=req.query, top_k=req.top_k),
            hits,
            cast(Callable[..., ResolveMeta], svc.resolve_meta),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/answer", response_model=AnswerResponse)
def answer(
    req: AnswerRequest, svc: SearchService = Depends(get_search_service)
) -> AnswerResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        dto_hits = [
            hit_to_search_hit(h, cast(Callable[..., ResolveMeta], svc.resolve_meta))
            for h in hits
        ]
        answer_text = " ".join(h.preview for h in dto_hits if h.preview) or "No answer."
        return AnswerResponse(query=req.query, answer=answer_text, sources=dto_hits)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
