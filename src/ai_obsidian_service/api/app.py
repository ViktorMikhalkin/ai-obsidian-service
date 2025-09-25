from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, cast

from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.api.schemas import InfoSchema, SearchRequest, SearchResponse
from ai_obsidian_service.di_selector import make_components
from ai_obsidian_service.domain.models import Query
from ai_obsidian_service.usecases.index_corpus import IndexCorpus

_rebuild_lock = asyncio.Lock()


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build iteration-5 components once, on startup.
        chunker = SimpleChunker(max_chars=1000, overlap=100)
        cmp = make_components(chunker=chunker)
        # Ensure explicit parser assignment for SearchService
        cmp.search.parser = MarkdownParser()
        app.state.components = cmp  # stored for dependency injection
        try:
            yield
        finally:
            # Optional shutdown hook
            try:
                cmp.search.shutdown()
            except Exception:
                pass

    return FastAPI(title="AI Obsidian Service", lifespan=lifespan)


app = create_app()


def get_search_service_dep(request: Request) -> SearchService:
    # Components are placed on app.state during lifespan
    cmp = cast(Any, request.app.state.components)
    svc: SearchService = cmp.search
    return svc


# Backward-compat alias some tests/tools still import
get_search_service = get_search_service_dep


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, svc: SearchService = Depends(get_search_service_dep)) -> SearchResponse:
    result = svc.search_text(req.query, top_k=req.top_k, collection=req.collection)
    return hits_to_search_response(
        Query(text=req.query, top_k=req.top_k),
        result.hits,
        # Resolver must accept a kw-only named parameter "chunk_id"
        lambda *, chunk_id: {},  # provide real meta lookup here if available
    )


@app.post("/index/rebuild")
async def index_rebuild(root: str, svc: SearchService = Depends(get_search_service_dep)):
    """
    Rebuild index for all files under 'root'. Returns 503 if a rebuild is in progress.
    """
    if _rebuild_lock.locked():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"code": "INDEX_REBUILDING", "message": "Rebuild in progress"},
        )

    async with _rebuild_lock:
        # Use the same chunker and a concrete parser; reuse the current service
        usecase = IndexCorpus(
            parsers=[MarkdownParser()],
            chunker=SimpleChunker(max_chars=1000, overlap=100),
            service=svc,
        )
        count = await asyncio.to_thread(usecase.run, root)
        return {"indexed": count}


@app.get("/info", response_model=InfoSchema)
def info(request: Request) -> InfoSchema:
    """
    Lightweight manifest: backend/model/dim/count/index_dir when available.
    """
    cmp = cast(Any, request.app.state.components)
    backend = (os.getenv("VECTOR_STORE_BACKEND") or cmp.store.__class__.__name__).lower()
    # best-effort probes; presence depends on concrete implementations
    model = getattr(cmp.embedder, "model_name", None)
    dim = getattr(cmp.embedder, "dim", None)
    count = getattr(cmp.store, "count", None)
    index_dir = getattr(cmp.store, "index_dir", None)
    return InfoSchema(backend=backend, model=model, dim=dim, count=count, index_dir=index_dir)
