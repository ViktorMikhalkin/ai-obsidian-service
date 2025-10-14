"""Health and info endpoints."""

from datetime import datetime

from fastapi import APIRouter

from ai_obsidian_service.api.dependencies import get_service
from ai_obsidian_service.api.endpoints.config import get_current_config
from ai_obsidian_service.api.logging import log_structured
from ai_obsidian_service.api.schemas import InfoSchema

router = APIRouter()


@router.get("/health")
def health_check():
    """Simple health check with device info."""
    device = "cpu"
    cuda_available = False
    try:
        import torch

        cuda_available = bool(
            getattr(torch, "cuda", None) and torch.cuda.is_available()
        )
        device = "cuda" if cuda_available else "cpu"
    except Exception:
        pass
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ai-obsidian-service",
        "device": device,
        "cuda_available": cuda_available,
        "version": "5.0-lite",
    }


@router.get("/info", response_model=InfoSchema)
def api_info() -> InfoSchema:
    """Get service information and current index statistics."""
    config = get_current_config()

    backend = config.indexing.backend or "memory"
    model = None
    dim = None
    count = None
    index_dir = config.indexing.index_dir

    try:
        if hasattr(get_service().index, "embedder") and hasattr(
            get_service().index.embedder, "model_name"
        ):
            model = get_service().index.embedder.model_name

        if hasattr(get_service().index, "store"):
            store = get_service().index.store
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
