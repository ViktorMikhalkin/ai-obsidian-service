from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8


class Citation(BaseModel):
    doc_path: str
    chunk_id: str
    snippet: str
    span: list[int] | None = None


class SearchHit(BaseModel):
    doc_path: str
    chunk_id: str
    score: float
    preview: str


class SearchResponse(BaseModel):
    hits: list[SearchHit]


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 8


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
