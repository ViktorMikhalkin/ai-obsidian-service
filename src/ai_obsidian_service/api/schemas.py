from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# ---- Requests ----

class SearchRequest(BaseModel):
    query: str = Field(..., description="User query text.")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return.")
    collection: str | None = Field(None, description="Optional collection (folder) to filter results.")


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
    preview: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHitDTO]


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHitDTO]


class InfoSchema(BaseModel):
    backend: str
    model: str | None = None
    dim: int | None = None
    count: int | None = None
    index_dir: str | None = None
