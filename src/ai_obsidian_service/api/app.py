"""Main FastAPI application."""

import warnings

from fastapi import FastAPI

from ai_obsidian_service.api.dependencies import RequestIdMiddleware, lifespan
from ai_obsidian_service.api.endpoints import health, indexing, ocr, search

warnings.filterwarnings(
    "ignore",
    message=".*resource_tracker.*",
    category=UserWarning,
    module="multiprocessing.resource_tracker",
)

app = FastAPI(
    title="AI Obsidian Service",
    version="5.0-lite",
    description="Local indexing & RAG API for Obsidian notes with OCR support and incremental indexing",
    lifespan=lifespan,
)

app.add_middleware(RequestIdMiddleware)

# Register routers
app.include_router(health.router, tags=["health"])
app.include_router(search.router, tags=["search"])
app.include_router(indexing.router, prefix="/index", tags=["indexing"])
app.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
