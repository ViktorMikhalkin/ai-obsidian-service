"""Health and info endpoints."""

import os
from datetime import datetime

from fastapi import APIRouter

from ai_obsidian_service.api.dependencies import _service, log_structured
from ai_obsidian_service.api.schemas import InfoSchema

router = APIRouter()


@router.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ai-obsidian-service",
        "version": "5.0-lite",
    }


@router.get("/info", response_model=InfoSchema)
def api_info() -> InfoSchema:
    """Get service information and current index statistics."""
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    model = None
    dim = None
    count = None
    index_dir = os.getenv("INDEX_DIR")

    try:
        if hasattr(_service.index, "embedder") and hasattr(
            _service.index.embedder, "model_name"
        ):
            model = _service.index.embedder.model_name

        if hasattr(_service.index, "store"):
            store = _service.index.store
            if hasattr(store, "dim"):
                dim = store.dim
            if hasattr(store, "count"):
                count = store.count
            if hasattr(store, "index_dir"):
                index_dir = store.index_dir or index_dir

        log_structured(
            "info", "info_retrieved", backend=backend, model=model, dim=dim, count=count
        )
    except Exception as e:
        log_structured("error", "info_retrieval_failed", error=str(e))

    return InfoSchema(
        backend=backend, model=model, dim=dim, count=count, index_dir=index_dir
    )
