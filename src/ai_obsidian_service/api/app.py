from __future__ import annotations
import os
from functools import lru_cache
from fastapi import FastAPI, Depends
from ai_obsidian_service import __version__
from ai_obsidian_service.config.container import build_search_service
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Query
from .schemas import SearchRequest, SearchResponse, AnswerRequest, AnswerResponse
from .mappers import hits_to_search_response, hit_to_search_hit

app = FastAPI(title="AI Obsidian Service", version=__version__)

@lru_cache(maxsize=1)
def get_search_service() -> SearchService:
    index_dir = os.getenv("AI_OBSIDIAN_INDEX_DIR", None)
    return build_search_service(index_dir=index_dir)

@app.get("/health")
def health():
    return {"ok": True, "version": __version__}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, svc: SearchService = Depends(get_search_service)) -> SearchResponse:
    hits = svc.search_text(req.query, top_k=req.top_k)
    return hits_to_search_response(Query(text=req.query, top_k=req.top_k), hits, svc.resolve_meta)

@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, svc: SearchService = Depends(get_search_service)) -> AnswerResponse:
    hits = svc.search_text(req.query, top_k=req.top_k)
    dto_hits = [hit_to_search_hit(h, svc.resolve_meta) for h in hits]
    answer_text = " ".join(h.preview for h in dto_hits if h.preview) or "No answer."
    return AnswerResponse(query=req.query, answer=answer_text, sources=dto_hits)
