from __future__ import annotations

import logging
from functools import lru_cache
from typing import Callable, cast
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, HTTPException

from .config.container import build_search_service
from .indexer.services import IndexerService  # alias of SearchService
from .api.schemas import SearchRequest, SearchResponse, AnswerRequest, AnswerResponse
from .api.mappers import hits_to_search_response, hit_to_search_hit, ResolveMeta
from .api.errors import install_error_handlers, ensure_request_id_middleware, install_access_logger
from .core import Query
from .logging_utils import configure_logging
from .config.settings import get_settings

settings = get_settings()

# Configure logging once (JSON + MDC filter)
level = getattr(logging, settings.log_level.upper(), logging.INFO)
configure_logging(
    json_enabled=settings.json_logs_enabled,
    level=level,
    log_uvicorn=settings.log_uvicorn,
)

@lru_cache(maxsize=1)
def get_indexer_service() -> IndexerService:
    index_dir = settings.index_dir
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

app = FastAPI(title=settings.app_name, lifespan=lifespan)
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
    logging.getLogger("ai_obsidian_service.app").info("health ping")
    return {"ok": True, "errors": []}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, svc: IndexerService = Depends(get_indexer_service)) -> SearchResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
        return hits_to_search_response(Query(text=req.query, top_k=req.top_k), hits, resolve)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, svc: IndexerService = Depends(get_indexer_service)) -> AnswerResponse:
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
