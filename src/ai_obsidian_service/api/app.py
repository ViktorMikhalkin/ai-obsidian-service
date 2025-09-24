
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from ai_obsidian_service import __version__
from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api.errors import install as install_error_stack
from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.api.schemas import (
    InfoSchema,
    SearchRequest,
    SearchResponse,
)
from ai_obsidian_service.config.settings import get_settings
from ai_obsidian_service.core import Query
from ai_obsidian_service.di_selector import make_components


def get_search_service_dep(request: Request) -> SearchService:
    svc: SearchService = request.app.state.components.search
    return svc

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Build components via DI selector
    components = make_components(chunker=SimpleChunker())
    app.state.components = components
    app.state.backend = settings.vector_store_backend
    app.state.index_dir = settings.vector_index_dir
    app.state.model_name = settings.st_model_name
    try:
        yield
    finally:
        # Optional: persist FAISS index if used (no-op for memory backend)
        backend = getattr(app.state, "backend", "memory")
        index_dir = getattr(app.state, "index_dir", None)
        model_name = getattr(app.state, "model_name", None)
        if backend == "faiss" and index_dir:
            try:
                components.store.save(index_dir, model_name=model_name)  # type: ignore[attr-defined]
                logging.getLogger("ai_obsidian_service.app").info(
                    "FAISS index saved",
                    extra={"path": index_dir, "model": model_name},
                )
            except Exception as e:  # pragma: no cover
                logging.getLogger("ai_obsidian_service.app").exception(
                    "Failed to save FAISS index on shutdown",
                    extra={"path": index_dir, "model": model_name, "error": str(e)},
                )

def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=__version__, lifespan=lifespan)
    # CORS
    origins = ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # errors and access logs
    install_error_stack(app)
    return app

app = create_app()

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, svc: SearchService = Depends(get_search_service_dep)) -> SearchResponse:
    res = svc.search_text(req.query, top_k=req.top_k, collection=req.collection)
    return hits_to_search_response(Query(text=req.query, top_k=req.top_k), res.hits, lambda _cid: {})

@app.get("/info", response_model=InfoSchema)
def info(request: Request) -> InfoSchema:
    components = request.app.state.components
    backend = getattr(request.app.state, "backend", "memory")
    index_dir = getattr(request.app.state, "index_dir", None)
    embedder = getattr(components.index, "embedder", None)
    store = getattr(components.index, "store", None)
    model = getattr(embedder, "model_name", None) or getattr(embedder, "__class__", type("X",(object,),{})).__name__
    dim = getattr(embedder, "dim", None)
    count = getattr(store, "count", lambda: None)()
    return InfoSchema(
        backend=backend,
        model=model,
        dim=dim if isinstance(dim, int) else None,
        count=count if isinstance(count, int) or count is None else None,
        index_dir=index_dir,
    )
