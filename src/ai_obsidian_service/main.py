from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import cast

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api.mappers import ResolveMeta, hit_to_search_hit, hits_to_search_response
from .api.schemas import AnswerRequest, AnswerResponse, SearchRequest, SearchResponse
from .config.container import build_search_service
from .indexer.services import IndexerService  # alias of SearchService


@lru_cache(maxsize=1)
def get_indexer_service() -> IndexerService:
    index_dir = os.getenv("AI_OBSIDIAN_INDEX_DIR", None)
    svc = build_search_service(index_dir=index_dir)
    return cast(IndexerService, svc)


def _status_code_to_code(status_code: int) -> str:
    mapping = {
        400: "bad_request",
        401: "unauthenticated",
        403: "forbidden",
        404: "not_found",
        409: "conflict",
        422: "validation_error",
        429: "rate_limited",
        500: "internal_error",
    }
    return mapping.get(status_code, f"http_{status_code}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    svc = get_indexer_service()
    try:
        yield
    finally:
        try:
            svc.shutdown()
        except Exception:
            # best-effort
            pass


app = FastAPI(title="AI Obsidian Service", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _http_exc_handler(request: Request, exc: HTTPException):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    payload = {
        "code": _status_code_to_code(exc.status_code),
        "message": str(exc.detail) if exc.detail else "HTTP error",
        "requestId": rid,
    }
    return JSONResponse(
        status_code=exc.status_code, content=payload, headers={"X-Request-ID": rid}
    )


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    payload = {
        "code": "validation_error",
        "message": "Validation failed",
        "requestId": rid,
    }
    return JSONResponse(status_code=422, content=payload, headers={"X-Request-ID": rid})


@app.exception_handler(Exception)
async def _unhandled_handler(request: Request, exc: Exception):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    payload = {
        "code": "internal_error",
        "message": "Internal Server Error",
        "requestId": rid,
    }
    return JSONResponse(status_code=500, content=payload, headers={"X-Request-ID": rid})


def _as_hits(obj):
    """Accept either List[Hit] or an object with .hits (e.g. SearchResult)."""
    try:
        return list(obj.hits)
    except AttributeError:
        return list(obj)


@app.get("/health")
def health():
    # tests expect exactly this payload
    return {"ok": True, "errors": []}


@app.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest, svc: IndexerService = Depends(get_indexer_service)
) -> SearchResponse:
    try:
        result = svc.search_text(req.query, req.top_k)
        hits = _as_hits(result)
        from .core import (
            Query,  # local import to avoid potential circular imports at type-check time
        )

        resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
        return hits_to_search_response(
            Query(text=req.query, top_k=req.top_k), hits, resolve
        )
    except Exception as e:
        # Let our exception handler format the payload uniformly
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/answer", response_model=AnswerResponse)
def answer(
    req: AnswerRequest, svc: IndexerService = Depends(get_indexer_service)
) -> AnswerResponse:
    result = svc.search_text(req.query, req.top_k)
    hits = _as_hits(result)
    resolve = cast(Callable[..., ResolveMeta], svc.resolve_meta)
    dto_hits = [hit_to_search_hit(h, resolve) for h in hits]
    answer_text = " ".join(h.preview for h in dto_hits if h.preview) or "No answer."
    return AnswerResponse(query=req.query, answer=answer_text, sources=dto_hits)
