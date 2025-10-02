from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import warnings
from collections.abc import Callable
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ai_obsidian_service.adapters.llm.ollama_client import OllamaClient
from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.api.schemas import (
    AnswerRequest,
    InfoSchema,
    SearchRequest,
)
from ai_obsidian_service.config.container import (
    build_index_corpus,
    build_search_service,
)
from ai_obsidian_service.domain.models import Query
from ai_obsidian_service.rag import answer_with_citations

# Suppress the multiprocessing resource tracker warning from ML libraries
warnings.filterwarnings(
    "ignore",
    message=".*resource_tracker.*",
    category=UserWarning,
    module="multiprocessing.resource_tracker",
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Configure structured logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,  # Override any existing configuration
)

# Set log level for our application
logging.getLogger("ai_obsidian_service").setLevel(logging.INFO)

# Reduce noise from other libraries
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Request ID tracking and structured logging
# -----------------------------------------------------------------------------

# Context variable for request tracking
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Adds request ID to all requests for log correlation"""

    async def dispatch(self, request: Request, call_next):
        # Use client-provided ID or generate new one
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request_id_ctx.set(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


def log_structured(level: str, message: str, **kwargs):
    """
    Structured logging helper that adds request_id automatically.

    Usage:
        log_structured("info", "search_completed", query="test", hits=5, latency_ms=42)
    """
    request_id = request_id_ctx.get("")

    # Build structured log message
    fields = {"request_id": request_id, **kwargs}
    field_str = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
    log_msg = f"{message} | {field_str}" if field_str else message

    # Log at appropriate level
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(log_msg)


# -----------------------------------------------------------------------------
# App + DI
# -----------------------------------------------------------------------------

# Core service (search/index)
_service = build_search_service(index_dir=os.getenv("INDEX_DIR"))


# Optional Ollama client (if env present)
def _make_ollama() -> OllamaClient | None:
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

# Lock for /index/rebuild
_rebuild_lock = asyncio.Lock()

# Resolve meta hook for mappers
_ResolveMeta = Callable[..., dict[str, Any]]


def _resolve_meta(chunk_id: str) -> dict[str, Any]:
    try:
        return _service.resolve_meta(chunk_id=chunk_id)
    except Exception:
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle with timeouts"""
    # Startup
    log_structured("info", "service_starting", version="5.0-lite")

    try:
        yield
    finally:
        # Shutdown: close resources gracefully with timeouts
        log_structured("info", "service_shutting_down")

        async def safe_shutdown(coro, name: str, timeout: float = 2.0):
            """Execute shutdown operation with timeout"""
            try:
                await asyncio.wait_for(coro, timeout=timeout)
                log_structured("info", "shutdown_completed", component=name)
            except TimeoutError:
                log_structured(
                    "warning", "shutdown_timeout", component=name, timeout_s=timeout
                )
            except Exception as e:
                log_structured("error", "shutdown_error", component=name, error=str(e))

        # Shutdown service
        if hasattr(_service, "shutdown"):
            await safe_shutdown(
                asyncio.to_thread(_service.shutdown), "Service", timeout=2.0
            )

        # Close index
        if hasattr(_service, "index") and hasattr(_service.index, "close"):
            await safe_shutdown(
                asyncio.to_thread(_service.index.close), "Index", timeout=2.0
            )

        # Close store
        if (
            hasattr(_service, "index")
            and hasattr(_service.index, "store")
            and hasattr(_service.index.store, "close")
        ):
            await safe_shutdown(
                asyncio.to_thread(_service.index.store.close), "Store", timeout=2.0
            )

        log_structured("info", "service_shutdown_complete")


app = FastAPI(
    title="AI Obsidian Service",
    version="5.0-lite",
    description="Local indexing & RAG API for Obsidian notes with real-time SSE progress",
    lifespan=lifespan,
)

# Register request ID middleware
app.add_middleware(RequestIdMiddleware)


# -----------------------------------------------------------------------------
# /index/rebuild - SSE streaming endpoint with batch processing
# -----------------------------------------------------------------------------


@app.post("/index/rebuild")
async def index_rebuild(root: str = Body(..., embed=True)):
    """
    Rebuild index from a root directory with real-time progress updates via SSE.

    Expects JSON body: {"root": "/path/to/dir"}

    Returns Server-Sent Events (SSE) stream with progress updates.

    Note: This endpoint cannot be tested in Swagger UI. Use curl:
    ```bash
    curl -N -X POST http://127.0.0.1:8000/index/rebuild \\
      -H "Content-Type: application/json" \\
      -H "Accept: text/event-stream" \\
      -d '{"root": "/path/to/vault"}'
    ```
    """

    # Validation
    if not root:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "MISSING_ROOT",
                "message": "Provide JSON body with 'root' field",
            },
        )

    p = Path(root)
    if not p.exists():
        log_structured(
            "warning", "index_rebuild_invalid_path", root=root, reason="not_found"
        )
        raise HTTPException(
            status_code=400,
            detail={"code": "ROOT_NOT_FOUND", "message": f"Path not found: {root}"},
        )
    if not p.is_dir():
        log_structured(
            "warning", "index_rebuild_invalid_path", root=root, reason="not_directory"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ROOT_NOT_DIR",
                "message": f"Path is not a directory: {root}",
            },
        )

    # Check if rebuild is already running
    if _rebuild_lock.locked():
        log_structured("warning", "index_rebuild_rejected", reason="already_running")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "code": "INDEX_REBUILDING",
                "message": "Rebuild already in progress",
            },
        )

    async def progress_stream():
        """Generate SSE progress events"""
        try:
            async with _rebuild_lock:
                # Phase 1: Scan directory
                log_structured("info", "index_rebuild_started", root=root)
                yield f"data: {json.dumps({'status': 'scanning', 'message': 'Scanning directory...'})}\n\n"

                # Get parsers to determine which files to index
                usecase = build_index_corpus(index_dir=os.getenv("INDEX_DIR"))
                file_paths = []

                for path in p.rglob("*"):
                    if not path.is_file():
                        continue
                    spath = str(path)
                    # Check if any parser can handle this file
                    for parser in usecase.parsers:
                        if parser.can_parse(spath):
                            file_paths.append(path)
                            break

                total = len(file_paths)
                log_structured("info", "files_scanned", total=total)

                if total == 0:
                    log_structured(
                        "info",
                        "index_rebuild_completed",
                        chunks=0,
                        files=0,
                        reason="no_files",
                    )
                    yield f"data: {json.dumps({'status': 'complete', 'indexed': 0, 'total': 0, 'message': 'No indexable files found'})}\n\n"
                    return

                yield f"data: {json.dumps({'status': 'started', 'total': total, 'message': f'Found {total} files to index'})}\n\n"

                # Phase 2: Index files with progress tracking and batch processing
                count = 0
                processed = 0
                errors = 0
                last_update = time.time()
                start_time = time.time()
                batch_size = 4  # Process 4 files concurrently

                async def process_file(file_path: Path) -> tuple[Path, int | None]:
                    """Process a single file and return (path, chunks_or_none)"""
                    try:
                        chunks = await asyncio.to_thread(
                            _service.index_path, str(file_path)
                        )
                        return (file_path, chunks)
                    except Exception as e:
                        log_structured(
                            "error",
                            "file_index_failed",
                            file=str(file_path),
                            error=str(e),
                        )
                        return (file_path, None)

                # Process files in batches
                for i in range(0, len(file_paths), batch_size):
                    batch = file_paths[i : i + batch_size]

                    # Process batch concurrently
                    results = await asyncio.gather(
                        *[process_file(fp) for fp in batch], return_exceptions=False
                    )

                    # Collect results
                    for _file_path, chunks in results:
                        processed += 1
                        if chunks is not None:
                            count += chunks
                        else:
                            errors += 1

                    # Send progress update every batch, every 2 seconds, or on last file
                    current_time = time.time()
                    should_update = (
                        current_time - last_update >= 2 or processed == total
                    )

                    if should_update:
                        elapsed = current_time - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        eta_seconds = (total - processed) / rate if rate > 0 else 0

                        # Get last file name from batch
                        current_file = batch[-1].name if batch else ""

                        progress_data = {
                            "status": "indexing",
                            "processed": processed,
                            "total": total,
                            "chunks": count,
                            "errors": errors,
                            "percent": int((processed / total) * 100),
                            "rate": round(rate, 2),
                            "eta_seconds": int(eta_seconds),
                            "current_file": current_file,
                        }
                        yield f"data: {json.dumps(progress_data)}\n\n"
                        last_update = current_time

                    # Keep-alive: send heartbeat every 15 seconds
                    if time.time() - last_update > 15:
                        yield ": keep-alive\n\n"
                        last_update = time.time()

                # Phase 3: Save index to disk
                log_structured("info", "index_saving")
                yield f"data: {json.dumps({'status': 'saving', 'message': 'Saving index to disk...'})}\n\n"

                try:
                    store = _service.index.store
                    if hasattr(store, "save"):
                        index_dir = os.getenv("INDEX_DIR")
                        if index_dir:
                            model_name = None
                            if hasattr(_service.index, "embedder") and hasattr(
                                _service.index.embedder, "model_name"
                            ):
                                model_name = _service.index.embedder.model_name

                            await asyncio.to_thread(
                                store.save, index_dir, model_name=model_name
                            )
                            log_structured("info", "index_saved", path=index_dir)
                        else:
                            log_structured(
                                "warning", "index_save_skipped", reason="no_index_dir"
                            )
                    else:
                        log_structured(
                            "warning", "index_save_skipped", reason="not_supported"
                        )
                except Exception as e:
                    log_structured("error", "index_save_failed", error=str(e))
                    logger.error(f"Failed to save index: {e}", exc_info=True)
                    yield f"data: {json.dumps({'status': 'save_error', 'error': str(e)})}\n\n"

                # Phase 4: Complete
                total_time = time.time() - start_time
                log_structured(
                    "info",
                    "index_rebuild_completed",
                    chunks=count,
                    files=processed,
                    errors=errors,
                    duration_s=round(total_time, 2),
                )

                final_data = {
                    "status": "complete",
                    "indexed": count,
                    "total": processed,
                    "errors": errors,
                    "total_time_seconds": round(total_time, 2),
                    "message": f"Indexed {count} chunks from {processed} files ({errors} errors)",
                }
                yield f"data: {json.dumps(final_data)}\n\n"

        except asyncio.CancelledError:
            log_structured("info", "index_rebuild_cancelled")
            yield f"data: {json.dumps({'status': 'cancelled', 'message': 'Rebuild cancelled'})}\n\n"
            raise
        except Exception as e:
            log_structured("error", "index_rebuild_failed", error=str(e))
            logger.exception("Index rebuild failed")
            yield f"data: {json.dumps({'status': 'failed', 'error': str(e)})}\n\n"

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# -----------------------------------------------------------------------------
# /search
# -----------------------------------------------------------------------------


@app.post("/search")
def api_search(req: SearchRequest):
    """
    Vector search with optional collection filter.

    Returns semantically similar chunks from the indexed corpus.
    """
    try:
        result = _service.search_text(
            req.query, top_k=req.top_k, collection=req.collection
        )

        # Ensure we have a valid query object
        query = result.query or Query(text=req.query, top_k=req.top_k)

        dto = hits_to_search_response(query, result.hits, _resolve_meta)

        log_structured(
            "info",
            "search_completed",
            query=req.query[:50],  # Truncate long queries
            hits=len(result.hits),
            latency_ms=result.total_time_ms,
        )

        return {
            "query": dto.query,
            "top_k": dto.top_k,
            "hits": [h.model_dump() for h in dto.hits],
            "retrieved_at": datetime.utcnow().isoformat() + "Z",
            "total_time_ms": result.total_time_ms,
        }
    except Exception as e:
        log_structured("error", "search_failed", query=req.query[:50], error=str(e))
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail={"code": "SEARCH_FAILED", "message": str(e)}
        ) from e


# -----------------------------------------------------------------------------
# /answer (RAG)
# -----------------------------------------------------------------------------


@app.post("/answer")
def api_answer(req: AnswerRequest):
    """
    Retrieve relevant chunks and generate an answer using LLM (RAG).

    Returns an AI-generated answer with citations to source chunks.
    """
    try:
        # 1) Retrieve relevant chunks
        result = _service.search_text(req.query, top_k=req.top_k)

        # 2) Generate answer using LLM
        system_prompt = os.getenv("OLLAMA_SYSTEM_PROMPT")
        text, _ = answer_with_citations(
            query=req.query,
            result=result,
            llm=_LLM,
            system_prompt=system_prompt,
        )

        # 3) Format citations
        citations: list[dict] = []
        for h in result.hits[:10]:
            # Get chunk text safely
            chunk_text = getattr(h, "chunk_text", None)
            if not chunk_text and getattr(h, "chunk", None) is not None:
                try:
                    chunk = h.chunk
                    if chunk is not None:
                        chunk_text = chunk.text
                except Exception:
                    chunk_text = None

            # Find snippet span in chunk text
            snippet = h.snippet or ""
            span = (-1, -1)
            if chunk_text:
                i = chunk_text.find(snippet)
                span = (i, i + len(snippet)) if i >= 0 and snippet else (-1, -1)

            # Extract document path from metadata
            doc_path = None
            try:
                chunk = getattr(h, "chunk", None)
                if chunk is not None and hasattr(chunk, "meta"):
                    meta = chunk.meta
                    if meta is not None:
                        doc_path = meta.get("path")
            except Exception:
                doc_path = None

            citations.append(
                {
                    "doc_path": doc_path,
                    "chunk_id": str(h.chunk_id),
                    "snippet": snippet,
                    "span": span,
                }
            )

        log_structured(
            "info",
            "answer_completed",
            query=req.query[:50],
            citations=len(citations),
        )

        return {
            "query": req.query,
            "answer": text.strip(),
            "citations": citations,
        }
    except Exception as e:
        log_structured("error", "answer_failed", query=req.query[:50], error=str(e))
        logger.error(f"Answer generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail={"code": "ANSWER_FAILED", "message": str(e)}
        ) from e


# -----------------------------------------------------------------------------
# /info
# -----------------------------------------------------------------------------


@app.get("/info", response_model=InfoSchema)
def api_info() -> InfoSchema:
    """
    Get service information and current index statistics.

    Returns backend type, model name, dimensions, chunk count, and index directory.
    """
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    model = None
    dim = None
    count = None
    index_dir = os.getenv("INDEX_DIR")

    try:
        # Get embedder info
        if hasattr(_service.index, "embedder") and hasattr(
            _service.index.embedder, "model_name"
        ):
            model = _service.index.embedder.model_name

        # Get store/index info
        if hasattr(_service.index, "store"):
            store = _service.index.store
            if hasattr(store, "dim"):
                dim = store.dim
            if hasattr(store, "count"):
                count = store.count
            if hasattr(store, "index_dir"):
                index_dir = store.index_dir or index_dir

        log_structured(
            "info",
            "info_retrieved",
            backend=backend,
            model=model,
            dim=dim,
            count=count,
        )
    except Exception as e:
        log_structured("error", "info_retrieval_failed", error=str(e))
        logger.error(f"Failed to retrieve service info: {e}")

    return InfoSchema(
        backend=backend, model=model, dim=dim, count=count, index_dir=index_dir
    )


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------


@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ai-obsidian-service",
        "version": "5.0-lite",
    }
