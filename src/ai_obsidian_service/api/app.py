from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, status, HTTPException, Body
from fastapi.responses import JSONResponse

from ai_obsidian_service.adapters.llm.ollama_client import OllamaClient
from ai_obsidian_service.api.mappers import (
    hits_to_search_response,
)
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
def _resolve_meta(chunk_id: str) -> dict[str, Any]:  # pragma: no cover
    try:
        return _service.resolve_meta(chunk_id=chunk_id)
    except Exception:
        return {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: all the things are built above
    try:
        yield
    finally:
        # shutdown: closes resources gracefully
        try:
            if hasattr(_service, "shutdown"):
                _service.shutdown()
        except Exception:
            pass
        try:
            if hasattr(_service, "index") and hasattr(_service.index, "close"):
                _service.index.close()
        except Exception:
            pass
        try:
            if hasattr(_service, "index") and hasattr(_service.index, "store") and hasattr(_service.index.store, "close"):
                _service.index.store.close()
        except Exception:
            pass

app = FastAPI(title="AI Obsidian Service", version="5.0-lite", lifespan=lifespan)


# -----------------------------------------------------------------------------
# /index/rebuild with lock (503 if already running)
# -----------------------------------------------------------------------------

@app.post("/index/rebuild")
async def index_rebuild(root: str = Body(..., embed=True)):
    """Rebuild index from a root directory. Expects JSON body: {"root": "/path/to/dir"}"""

    if not root:
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_ROOT", "message": "Provide JSON body with 'root' field"}
        )

    p = Path(root)
    if not p.exists():
        raise HTTPException(
            status_code=400,
            detail={"code": "ROOT_NOT_FOUND", "message": f"Path not found: {root}"}
        )
    if not p.is_dir():
        raise HTTPException(
            status_code=400,
            detail={"code": "ROOT_NOT_DIR", "message": f"Path is not a directory: {root}"}
        )

    if _rebuild_lock.locked():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"code": "INDEX_REBUILDING", "message": "Rebuild in progress"},
        )

    async with _rebuild_lock:
        try:
            usecase = build_index_corpus(index_dir=os.getenv("INDEX_DIR"))
            count = await asyncio.to_thread(usecase.run, str(p))
            return {"indexed": count, "root": str(p)}
        except Exception as e:
            import traceback
            return JSONResponse(
                status_code=500,
                content={"code": "REBUILD_FAILED", "message": str(e), "trace": traceback.format_exc()},
            )


# -----------------------------------------------------------------------------
# /search
# -----------------------------------------------------------------------------

@app.post("/search")
def api_search(req: SearchRequest):
    """
    Vector search with optional collection filter and (opt.) rerank.
    """
    result = _service.search_text(req.query, top_k=req.top_k, collection=req.collection)

    # Ensure we have a valid query object
    query = result.query or Query(text=req.query, top_k=req.top_k)

    dto = hits_to_search_response(query, result.hits, _resolve_meta)
    return {
        "query": dto.query,
        "top_k": dto.top_k,
        "hits": [h.model_dump() for h in dto.hits],
        "retrieved_at": datetime.utcnow().isoformat() + "Z",
        "total_time_ms": result.total_time_ms,
    }


# -----------------------------------------------------------------------------
# /answer  (RAG over /search + optional Ollama)
# -----------------------------------------------------------------------------

@app.post("/answer")
def api_answer(req: AnswerRequest):
    """
    Retrieve → (compose context) → Ask LLM (Ollama) → Cite (compat format).
    """
    # 1) retrieve
    result = _service.search_text(req.query, top_k=req.top_k)

    # 2) текст ответа (через Ollama или мини-фолбек)
    system_prompt = os.getenv("OLLAMA_SYSTEM_PROMPT")
    text, _ = answer_with_citations(
        query=req.query,
        result=result,
        llm=_LLM,
        system_prompt=system_prompt,
    )

    # 3) цитаты в «минимальной» форме, как ждут тесты
    citations: list[dict] = []
    for h in result.hits[:10]:
        # get chunk text safely
        chunk_text = getattr(h, "chunk_text", None)
        if not chunk_text and getattr(h, "chunk", None) is not None:
            try:
                chunk = h.chunk
                if chunk is not None:
                    chunk_text = chunk.text
            except Exception:
                chunk_text = None

        # span: search for snippet in the input chunk text (if exists)
        snippet = h.snippet or ""
        span = (-1, -1)
        if chunk_text:
            i = chunk_text.find(snippet)
            span = (i, i + len(snippet)) if i >= 0 and snippet else (-1, -1)

        # path from metadata
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

    return {
        "query": req.query,
        "answer": text.strip(),
        "citations": citations,
    }

# -----------------------------------------------------------------------------
# /info
# -----------------------------------------------------------------------------

@app.get("/info", response_model=InfoSchema)
def api_info() -> InfoSchema:
    """
    Minimal manifest of the current backend.
    """
    backend = os.getenv("VECTOR_STORE_BACKEND", "memory")
    model = None
    dim = None
    count = None
    index_dir = os.getenv("INDEX_DIR")

    try:
        # embedder info
        if hasattr(_service.index, "embedder") and hasattr(_service.index.embedder, "model_name"):
            model = _service.index.embedder.model_name
        # store/index info
        if hasattr(_service.index, "store") and hasattr(_service.index.store, "dim"):
            dim = _service.index.store.dim
        if hasattr(_service.index, "store") and hasattr(_service.index.store, "count"):
            count = _service.index.store.count
        if hasattr(_service.index, "store") and hasattr(_service.index.store, "index_dir"):
            index_dir = _service.index.store.index_dir or index_dir
    except Exception:
        pass

    return InfoSchema(backend=backend, model=model, dim=dim, count=count, index_dir=index_dir)