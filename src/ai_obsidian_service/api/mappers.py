from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, TypedDict

from ai_obsidian_service.core import ChunkId, DocId, Hit, Query

from .schemas import AnswerResponse, SearchHit, SearchResponse


class ResolveMeta(TypedDict, total=False):
    path: str
    kind: str
    preview: str
    text: str
    score: float


# Resolver may support multiple signatures; keep it variadic.
ResolveFn = Callable[..., ResolveMeta | Mapping[str, Any] | Any]


def _call_resolve(
    resolve: ResolveFn, doc_id: DocId, chunk_id: ChunkId, order: int
) -> Any:
    # Prefer the modern signature (doc_id, chunk_id, order).
    try:
        return resolve(doc_id, chunk_id, order)
    except TypeError:
        # Fallback to legacy (doc_id, order).
        try:
            return resolve(doc_id, order)
        except TypeError:
            # Last resort: just (doc_id).
            return resolve(doc_id)


def _normalize_meta(meta: Any) -> dict[str, Any]:
    """Make sure meta is a plain dict. Be resilient to mocks/objects."""
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, Mapping):
        return dict(meta)
    # Try attribute access (works for SimpleNamespace / Mock with configured attrs)
    result: dict[str, Any] = {}
    for key in ("path", "kind", "preview", "text", "score"):
        if hasattr(meta, key):
            try:
                result[key] = getattr(meta, key)
            except Exception:
                pass
    return result


def _to_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return default


def hit_to_search_hit(hit: Hit, resolve: ResolveFn) -> SearchHit:
    raw = _call_resolve(resolve, hit.doc_id, hit.chunk_id, hit.chunk_order)
    meta = _normalize_meta(raw)
    path = _to_str(meta.get("path"), "")
    kind = _to_str(meta.get("kind"), "chunk")
    preview_raw = meta.get("preview") or meta.get("text") or ""
    preview = _to_str(preview_raw, "")[:240]
    return SearchHit(
        id=str(hit.chunk_id),  # DTO id must be string
        path=path,
        kind=kind,
        preview=preview,
        score=float(hit.score),
    )


def hits_to_search_response(
    query: Query, hits: list[Hit], resolve: ResolveFn
) -> SearchResponse:
    return SearchResponse(results=[hit_to_search_hit(h, resolve) for h in hits])


def answer_to_answer_response(
    query_text: str, answer_text: str, sources: list[SearchHit]
) -> AnswerResponse:
    return AnswerResponse(query=query_text, answer=answer_text, sources=sources)
