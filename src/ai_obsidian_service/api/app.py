from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, status
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
def _resolve_meta(*, chunk_id: str) -> dict[str, Any]:  # pragma: no cover
    try:
        return _service.resolve_meta(chunk_id)
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
async def index_rebuild(root: str):
    if _rebuild_lock.locked():
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"code": "INDEX_REBUILDING", "message": "Rebuild in progress"},
        )
    async with _rebuild_lock:
        usecase = build_index_corpus(index_dir=os.getenv("INDEX_DIR"))
        count = await asyncio.to_thread(usecase.run, root)
        return {"indexed": count, "root": root}


# -----------------------------------------------------------------------------
# /search
# -----------------------------------------------------------------------------

@app.post("/search")
def api_search(req: SearchRequest):
    """
    Vector search with optional collection filter and (opt.) rerank.
    """
    result = _service.search_text(req.query, top_k=req.top_k, collection=req.collection)
    dto = hits_to_search_response(result.query, result.hits, _resolve_meta)
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
                chunk_text = h.chunk.text
            except Exception:
                chunk_text = None

        # span: search for snippet in the input chunk text (if exists)
        snippet = h.snippet or ""
        span = (-1, -1)
        if chunk_text:
            i = chunk_text.find(snippet)
            span = (i, i + len(snippet)) if i >= 0 and snippet else (-1, -1)

        # path form metadata
        doc_path = None
        try:
            if getattr(h, "chunk", None) is not None and hasattr(h.chunk, "meta"):
                doc_path = h.chunk.meta.get("path")
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
