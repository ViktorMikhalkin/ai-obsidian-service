from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from functools import cast
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request

from ai_obsidian_service import __version__
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api.errors import install as install_error_stack
from ai_obsidian_service.config.settings import get_settings
from ai_obsidian_service.core import Query
from ai_obsidian_service.di_selector import make_components
from ai_obsidian_service.logging_utils import install_json_logging
from ai_obsidian_service.simple_chunker import SimpleChunker  # <<— ваш Chunker

from .mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse

# ------------------------------------------------------------------------------
# Settings & logging
# ------------------------------------------------------------------------------

settings = get_settings()

level = getattr(logging, settings.log_level.upper(), logging.INFO)
if settings.json_logs_enabled:
    install_json_logging(
        level=level,
        logger_names=["ai_obsidian_service.api.access"],
        include_uvicorn=settings.log_uvicorn,
    )
else:
    logging.basicConfig(level=level)

# ------------------------------------------------------------------------------
# App wiring (DI v2)
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Select backend via env:
    #   VECTOR_STORE_BACKEND = "memory" | "faiss"
    #   ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    app.state.components = make_components(chunker=SimpleChunker())
    try:
        yield
    finally:
        # place for graceful shutdown if needed (eg. store.flush())
        pass


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)

# unified requestId/handlers/access log are installed in one place
install_error_stack(app)


def _as_hits(obj):
    try:
        return list(obj.hits)
    except AttributeError:
        return list(obj)


def get_search_service_dep(request: Request) -> SearchService:
    # DI: retrieve SearchService from app.state (assembled in lifespan)
    return cast(SearchService, request.app.state.components.search)

# ------------------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------------------

@app.get("/health")
def health(_: Request):
    logging.getLogger("ai_obsidian_service.app").info("health ping")
    return {"ok": True, "errors": []}


@app.post("/search", response_model=SearchResponse)
def search(
        req: SearchRequest, svc: SearchService = Depends(get_search_service_dep)
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
        req: AnswerRequest, svc: SearchService = Depends(get_search_service_dep)
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
