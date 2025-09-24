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
from ai_obsidian_service.simple_chunker import SimpleChunker
from ai_obsidian_service.index.faiss_store import FaissVectorStore  # for save/load

from .mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse, IndexRequest

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
# App wiring (DI v2) + FAISS persistence via settings.vector_*
# ------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Assemble components via DI and (optionally) persist FAISS index.

    Settings (env-backed, see config/settings.py):
      - settings.vector_store_backend: "memory" | "faiss"
      - settings.st_model_name: SentenceTransformers model name
      - settings.vector_index_dir: directory for FAISS index (enable load/save)
    """
    backend = settings.vector_store_backend.lower()
    model_name = settings.st_model_name
    index_dir = settings.vector_index_dir

    # Build components (store may be in-memory or faiss)
    components = make_components(chunker=SimpleChunker(), backend=backend, model_name=model_name)

    # If faiss + index_dir provided → replace empty store with a loaded one
    if backend == "faiss" and index_dir:
        try:
            loaded_store = FaissVectorStore.load(index_dir, expected_model_name=model_name)
            components.store = loaded_store
            components.index.store = loaded_store  # rebind in the facade
            logging.getLogger("ai_obsidian_service.app").info(
                "FAISS index loaded",
                extra={"path": index_dir, "model": model_name},
            )
        except FileNotFoundError:
            # No existing index — start fresh; will save on shutdown
            logging.getLogger("ai_obsidian_service.app").warning(
                "FAISS index path not found, starting with an empty store",
                extra={"path": index_dir, "model": model_name},
            )

    app.state.components = components
    try:
        yield
    finally:
        # On shutdown: persist FAISS index if configured
        if backend == "faiss" and index_dir:
            try:
                components.store.save(index_dir, model_name=model_name)  # type: ignore[attr-defined]
                logging.getLogger("ai_obsidian_service.app").info(
                    "FAISS index saved",
                    extra={"path": index_dir, "model": model_name},
                )
            except Exception as e:
                logging.getLogger("ai_obsidian_service.app").exception(
                    "Failed to save FAISS index on shutdown",
                    extra={"path": index_dir, "model": model_name, "error": str(e)},
                )


app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)

# Unified requestId / access log / error handlers
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
def health(_: Request) -> dict:
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


@app.post("/index")
def index_file(
        req: IndexRequest, svc: SearchService = Depends(get_search_service_dep)
) -> dict:
    try:
        n = svc.index_path(req.path)
        return {"indexed_chunks": n}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
