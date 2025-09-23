from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)

class SearchHit(BaseModel):
    id: str
    path: str
    kind: str = "chunk"
    preview: str = ""
    score: float = 0.0

class SearchResponse(BaseModel):
    results: list[SearchHit] = []

class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5

class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHit] = []
