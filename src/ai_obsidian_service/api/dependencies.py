"""Shared dependencies and service initialization."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from ai_obsidian_service.adapters.llm.ollama_client import OllamaClient
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

logger = logging.getLogger(__name__)

# Request tracking
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Adds request ID to all requests for log correlation."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_ctx.set(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def log_structured(level: str, message: str, **kwargs):
    """Structured logging helper that adds request_id automatically."""
    request_id = request_id_ctx.get("")
    fields = {"request_id": request_id, **kwargs}
    field_str = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    log_msg = f"{message} | {field_str}" if field_str else message
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(log_msg)


# Service initialization
_service = build_search_service(index_dir=os.getenv("INDEX_DIR"))


def _make_ollama() -> OllamaClient | None:
    """Initialize Ollama client if configured."""
    base = os.getenv("OLLAMA_BASE_URL")
    model = os.getenv("OLLAMA_MODEL")
    if not base or not model:
        return None
    timeout_s = float(os.getenv("OLLAMA_TIMEOUT", "30"))
    try:
        return OllamaClient(base_url=base, model=model, timeout_s=timeout_s)
    except Exception:
        return None


_LLM = _make_ollama()

# Locks for long-running operations
_rebuild_lock = asyncio.Lock()
_ocr_lock = asyncio.Lock()


def resolve_meta(chunk_id: str) -> dict[str, Any]:
    """Resolve chunk metadata."""
    try:
        return _service.resolve_meta(chunk_id=chunk_id)
    except Exception:
        return {}


@asynccontextmanager
async def lifespan(app):
    """Manage application lifecycle with graceful shutdown."""
    log_structured("info", "service_starting", version="5.0-lite")

    # Start background task to load index if needed
    load_task = None
    if (
        hasattr(_service.index, "_pending_load_path")
        and _service.index._pending_load_path
    ):
        index_dir = _service.index._pending_load_path
        model_name = getattr(_service.index, "_pending_load_model", None)

        async def load_index_background():
            """Load FAISS index in background without blocking startup."""
            try:
                log_structured("info", "index_load_started", path=index_dir)

                # Load in thread pool to avoid blocking
                loaded_store = await asyncio.to_thread(
                    FaissVectorStore.load, index_dir, expected_model_name=model_name
                )

                # Replace the empty store with loaded one
                _service.index.store = loaded_store

                # Load registry if using EnhancedEmbeddingIndex
                if hasattr(_service.index, "load_registry"):
                    registry_path = Path(index_dir) / "doc_registry.json"
                    if registry_path.exists():
                        await asyncio.to_thread(
                            _service.index.load_registry, registry_path
                        )
                        log_structured(
                            "info",
                            "registry_loaded",
                            documents=len(_service.index._doc_registry),
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

        if hasattr(_service, "shutdown"):
            await safe_shutdown(
                asyncio.to_thread(_service.shutdown), "Service", timeout=2.0
            )

        if hasattr(_service, "index") and hasattr(_service.index, "close"):
            await safe_shutdown(
                asyncio.to_thread(_service.index.close), "Index", timeout=2.0
            )

        if (
            hasattr(_service, "index")
            and hasattr(_service.index, "store")
            and hasattr(_service.index.store, "close")
        ):
            await safe_shutdown(
                asyncio.to_thread(_service.index.store.close), "Store", timeout=2.0
            )

        log_structured("info", "service_shutdown_complete")
