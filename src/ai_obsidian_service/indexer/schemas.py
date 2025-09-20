from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict | None = None


class SearchHit(BaseModel):
    id: str  # Changed from int to support rich IDs like "doc123-chunk5"
    path: str
    kind: str
    preview: str
    score: float
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict | None = None


class SearchResponse(BaseModel):
    results: list[SearchHit]
    total_time_ms: float | None = None
    retrieved_at: str | None = None


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "auto"
    model: str | None = None
    max_tokens: int = 256
    temperature: float = 0.2


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHit]
    total_time_ms: float | None = None
