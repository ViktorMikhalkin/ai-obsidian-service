from pydantic import BaseModel
from typing import List, Optional


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class Citation(BaseModel):
    doc_path: str
    chunk_id: str
    snippet: str
    span: Optional[List[int]] = None


class SearchHit(BaseModel):
    doc_path: str
    chunk_id: str
    score: float
    preview: str


class SearchResponse(BaseModel):
    hits: List[SearchHit]


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 8


class AnswerResponse(BaseModel):
    answer: str
    citations: List[Citation]
