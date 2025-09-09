from fastapi import FastAPI
from .models import SearchRequest, SearchResponse, AnswerRequest, AnswerResponse
from .search import search
from .rag import answer_with_citations

app = FastAPI(title="AI↔Obsidian Indexer", version="0.1.0", description="Local indexing & RAG API for Obsidian + CLI")

@app.get("/health", summary="Health check", tags=["meta"])
def health():
    return {"status": "ok"}

@app.post("/search", response_model=SearchResponse, summary="Full-text search", tags=["search"])
def api_search(payload: SearchRequest):
    hits = search(payload.query, payload.top_k)
    return SearchResponse(hits=hits)

@app.post("/answer", response_model=AnswerResponse, summary="Answer with citations (Ollama required)", tags=["rag"])
def api_answer(payload: AnswerRequest):
    hits = search(payload.query, payload.top_k)
    ans, cites = answer_with_citations(payload.query, hits)
    return AnswerResponse(answer=ans, citations=cites)


from .index_admin import get_index_stats, rebuild_index
from pydantic import BaseModel
from typing import Optional, Dict, Any

class RebuildRequest(BaseModel):
    timeout_sec: Optional[int] = 0

@app.get("/index/stats", summary="Index statistics", tags=["index"])
def api_index_stats():
    stats = get_index_stats()
    return stats.__dict__

@app.post("/index/rebuild", summary="Rebuild index", tags=["index"])
def api_index_rebuild(payload: RebuildRequest):
    info = rebuild_index(timeout_sec=payload.timeout_sec or 0)
    return {"status": "started", **info}


# --- Enhanced route metadata for OpenAPI docs ---
from .models import SearchRequest, SearchResponse, AnswerRequest, AnswerResponse
from .rag import answer_with_citations
from .search import search

@app.get("/health", summary="Health check", tags=["meta"])
def health_route():
    return {"status": "ok"}

@app.post("/search",
          response_model=SearchResponse,
          summary="Full-text search",
          tags=["search"])
def api_search_with_meta(payload: SearchRequest):
    return search(payload)

@app.post("/answer",
          response_model=AnswerResponse,
          summary="Answer with citations (Ollama required)",
          tags=["rag"])
def api_answer_with_meta(payload: AnswerRequest):
    return answer_with_citations(payload)

@app.get("/index/stats",
         summary="Index statistics",
         tags=["index"])
def api_index_stats_with_meta():
    stats = get_index_stats()
    return stats.__dict__

@app.post("/index/rebuild",
          summary="Rebuild index",
          tags=["index"])
def api_index_rebuild_with_meta(payload: RebuildRequest):
    info = rebuild_index(timeout_sec=payload.timeout_sec or 0)
    return {"status": "started", **info}
