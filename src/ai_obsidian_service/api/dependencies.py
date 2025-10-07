"""Shared dependencies and service initialization."""

from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ai_obsidian_service.adapters.llm.ollama_client import OllamaClient
from ai_obsidian_service.api.endpoints.config import get_current_config
from ai_obsidian_service.api.logging import log_structured, request_id_ctx
from ai_obsidian_service.config.container import build_search_service
from ai_obsidian_service.index.faiss_store import FaissVectorStore

# Logging setup with file output
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ai_obsidian_service.log"),
    ],
    force=True,
)

logging.getLogger("ai_obsidian_service").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Adds request ID to all requests for log correlation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


# Service initialization - lazy loaded to avoid import-time crashes in tests
_service = None


def _init_service():
    """Initialize service using configuration."""
    global _service
    if _service is not None:
        return _service

    config = get_current_config()
    index_dir = config.indexing.index_dir if config.indexing.index_dir else None
    _service = build_search_service(index_dir=index_dir)
    return _service


def get_service():
    """Get the service instance, initializing if needed."""
    return _init_service()


def _make_ollama() -> OllamaClient | None:
    """Initialize Ollama client from configuration."""
    config = get_current_config()

    if not config.ollama_base_url or not config.ollama_model:
        return None

    try:
        return OllamaClient(
            base_url=config.ollama_base_url,
            model=config.ollama_model,
            timeout_s=config.ollama_timeout,
        )
    except Exception:
        return None


_LLM = _make_ollama()

# Locks for long-running operations
_rebuild_lock = asyncio.Lock()
_ocr_lock = asyncio.Lock()


def resolve_meta(chunk_id: str) -> dict[str, Any]:
    """Resolve chunk metadata."""
    try:
        result = get_service().resolve_meta(chunk_id=chunk_id)  # Changed
        if isinstance(result, dict):
            return result
        return {}
    except Exception:
        return {}


@asynccontextmanager
async def lifespan(app):
    """Manage application lifecycle with graceful shutdown."""
    log_structured("info", "service_starting", version="5.0-lite")

    # Start background task to load index if needed
    load_task = None
    if (
        hasattr(get_service().index, "_pending_load_path")
        and get_service().index._pending_load_path
    ):
        index_dir = get_service().index._pending_load_path
        model_name = getattr(get_service().index, "_pending_load_model", None)

        async def load_index_background():
            """Load FAISS index in background without blocking startup."""
            try:
                log_structured("info", "index_load_started", path=index_dir)

                # Load in thread pool to avoid blocking
                loaded_store = await asyncio.to_thread(
                    FaissVectorStore.load, index_dir, expected_model_name=model_name
                )

                # Replace the empty store with loaded one
                get_service().index.store = loaded_store

                # Load registry if using EnhancedEmbeddingIndex
                if hasattr(get_service().index, "load_registry"):
                    registry_path = Path(index_dir) / "doc_registry.json"
                    if registry_path.exists():
                        await asyncio.to_thread(
                            get_service().index.load_registry, registry_path
                        )
                        log_structured(
                            "info",
                            "registry_loaded",
                            documents=len(get_service().index._doc_registry),
                        )

                log_structured(
                    "info",
                    "index_load_complete",
                    chunks=loaded_store.count,
                    path=index_dir,
                )

            except Exception as e:
                log_structured(
                    "error", "index_load_failed", error=str(e), path=index_dir
                )

        # Start background loading
        load_task = asyncio.create_task(load_index_background())
        log_structured("info", "index_loading_background", path=index_dir)

    try:
        yield
    finally:
        log_structured("info", "service_shutting_down")

        # Cancel background task if still running
        if load_task and not load_task.done():
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                pass

        async def safe_shutdown(coro, name: str, timeout: float = 2.0):
            try:
                await asyncio.wait_for(coro, timeout=timeout)
                log_structured("info", "shutdown_completed", component=name)
            except TimeoutError:
                log_structured(
                    "warning", "shutdown_timeout", component=name, timeout_s=timeout
                )
            except Exception as e:
                log_structured("error", "shutdown_error", component=name, error=str(e))

        if hasattr(get_service(), "shutdown"):
            await safe_shutdown(
                asyncio.to_thread(get_service().shutdown), "Service", timeout=2.0
            )

        if hasattr(get_service(), "index") and hasattr(get_service().index, "close"):
            await safe_shutdown(
                asyncio.to_thread(get_service().index.close), "Index", timeout=2.0
            )

        if (
            hasattr(get_service(), "index")
            and hasattr(get_service().index, "store")
            and hasattr(get_service().index.store, "close")
        ):
            await safe_shutdown(
                asyncio.to_thread(get_service().index.store.close), "Store", timeout=2.0
            )

        log_structured("info", "service_shutdown_complete")
