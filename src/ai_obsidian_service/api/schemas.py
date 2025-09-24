from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional


# ---- Requests ----

class SearchRequest(BaseModel):
    query: str = Field(..., description="User query text.")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return.")


class AnswerRequest(BaseModel):
    query: str = Field(..., description="Question to answer using RAG over the index.")
    top_k: int = Field(5, ge=1, le=20, description="How many passages to consider.")


class IndexRequest(BaseModel):
    path: str = Field(..., description="Filesystem path to a document to index.")


# ---- Responses ----

class SearchHitDTO(BaseModel):
    id: str
    score: float
    text: str
    preview: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: List[SearchHitDTO]


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: List[SearchHitDTO]
