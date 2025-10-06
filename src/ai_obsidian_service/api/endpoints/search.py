"""Search and answer (RAG) endpoints."""

from datetime import datetime

from fastapi import APIRouter, HTTPException

from ai_obsidian_service.api.dependencies import (
    _LLM,
    _service,
    log_structured,
    logger,
    resolve_meta,
)
from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.api.schemas import AnswerRequest, SearchRequest
from ai_obsidian_service.domain.models import Query
from ai_obsidian_service.rag import answer_with_citations
from ai_obsidian_service.utils.config_helpers import ConfigValidator

router = APIRouter()


@router.post("/search")
def api_search(req: SearchRequest):
    """Vector search with optional collection filter."""

    validator = ConfigValidator()
    validator.require_search()

    try:
        result = _service.search_text(
            req.query, top_k=req.top_k, collection=req.collection
        )

        query = result.query or Query(text=req.query, top_k=req.top_k)
        dto = hits_to_search_response(query, result.hits, resolve_meta)

        log_structured(
            "info",
            "search_completed",
            query=req.query[:50],
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


@router.post("/answer")
def api_answer(req: AnswerRequest):
    """Retrieve relevant chunks and generate an answer using LLM (RAG)."""

    validator = ConfigValidator()
    _config = validator.require_rag()

    try:
        result = _service.search_text(req.query, top_k=req.top_k)

        system_prompt = None
        text, _ = answer_with_citations(
            query=req.query,
            result=result,
            llm=_LLM,
            system_prompt=system_prompt,
        )

        citations: list[dict] = []
        for h in result.hits[:10]:
            chunk_text = getattr(h, "chunk_text", None)
            if not chunk_text and getattr(h, "chunk", None) is not None:
                try:
                    chunk = h.chunk
                    if chunk is not None:
                        chunk_text = chunk.text
                except Exception:
                    chunk_text = None

            snippet = h.snippet or ""
            span = (-1, -1)
            if chunk_text:
                i = chunk_text.find(snippet)
                span = (i, i + len(snippet)) if i >= 0 and snippet else (-1, -1)

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
