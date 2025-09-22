from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import cast

from fastapi import Depends, FastAPI, HTTPException, Request

from .api.errors import (
    ensure_request_id_middleware,
    install_access_logger,
    install_error_handlers,
)
from .api.mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .api.schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse
from .config.container import build_search_service
from .core import Query
from .indexer.services import IndexerService  # alias of SearchService


@lru_cache(maxsize=1)
def get_indexer_service() -> IndexerService:
    index_dir = os.getenv("AI_OBSIDIAN_INDEX_DIR", None)
    svc = build_search_service(index_dir=index_dir)
    return cast(IndexerService, svc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = get_indexer_service()
    try:
        yield
    finally:
        try:
            svc.shutdown()
        except Exception:
            pass


app = FastAPI(title="AI Obsidian Service", lifespan=lifespan)
ensure_request_id_middleware(app)
install_access_logger(app)
install_error_handlers(app)


def _as_hits(obj):
    try:
        return list(obj.hits)
    except AttributeError:
        return list(obj)


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
    req: SearchRequest, svc: IndexerService = Depends(get_indexer_service)
) -> SearchResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
        return hits_to_search_response(
            Query(text=req.query, top_k=req.top_k), hits, resolve
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/answer", response_model=AnswerResponse)
def answer(
    req: AnswerRequest, svc: IndexerService = Depends(get_indexer_service)
) -> AnswerResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
        dto_hits = [hit_to_search_hit(h, resolve) for h in hits]
        answer_text = " ".join(h.preview for h in dto_hits if h.preview) or "No answer."
        return AnswerResponse(query=req.query, answer=answer_text, sources=dto_hits)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
