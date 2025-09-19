from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchHit(BaseModel):
    id: int
    path: str
    kind: str
    preview: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchHit]


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
