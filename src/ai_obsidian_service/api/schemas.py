from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="User query text.")
    top_k: int = Field(5, ge=1, le=50)
    collection: str | None = None

class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchHitDTO(BaseModel):
    id: str
    score: float
    text: str
    preview: str | None = None
    meta: dict[str, Any] | None = None
    # совместимость
    path: str | None = None
    kind: str | None = None
    doc_path: str | None = None
    chunk_id: str | None = None

class SearchResponse(BaseModel):
    query: str
    top_k: int
    hits: list[SearchHitDTO]
    @property
    def results(self) -> list[SearchHitDTO]:
        return self.hits

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
