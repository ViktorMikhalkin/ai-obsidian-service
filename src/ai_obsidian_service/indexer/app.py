from functools import lru_cache

from fastapi import Depends, FastAPI

from .schemas import (
    AnswerRequest,
    AnswerResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from .services import IndexerService


@lru_cache(maxsize=1)
def get_indexer_service():
    # In production, consider caching this instance for better performance
    return IndexerService()


app = FastAPI()


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    service: IndexerService = Depends(get_indexer_service),
):
    # 1. Embed the query
    query_embedding = service.embed.embed([request.query])[0]
    # 2. Perform search in the vector index
    top_indices, top_scores = service.fa.search(query_embedding, k=request.top_k)
    results = []
    for idx, score in zip(top_indices, top_scores, strict=False):
        if idx < len(service.metas):
            meta = service.metas[idx]
            results.append(
                SearchHit(
                    id=idx,
                    path=meta.get("path", ""),
                    kind=meta.get("kind", ""),
                    preview=meta.get("preview", meta.get("text", "")[:200]),
                    score=float(score),
                )
            )
    return SearchResponse(results=results)


@app.post("/answer", response_model=AnswerResponse)
async def answer(
    request: AnswerRequest,
    service: IndexerService = Depends(get_indexer_service),
):
    # Step 1: Embed query and retrieve top_k passages
    query_embedding = service.embed.embed([request.query])[0]
    top_indices, top_scores = service.fa.search(query_embedding, k=request.top_k)
    sources = []
    context_chunks = []
    for idx, score in zip(top_indices, top_scores, strict=False):
        if idx < len(service.metas):
            meta = service.metas[idx]
            sources.append(
                SearchHit(
                    id=idx,
                    path=meta.get("path", ""),
                    kind=meta.get("kind", ""),
                    preview=meta.get("preview", meta.get("text", "")[:200]),
                    score=float(score),
                )
            )
            # Collect context for extractive answer
            context_chunks.append(meta.get("text", ""))

    # Step 2: Generate answer (extractive mode for now)
    # For "auto" mode, you can plug in an LLM call here if available
    answer_text = ""
    if context_chunks:
        # Simple extractive: return the most relevant chunk (could be improved)
        answer_text = context_chunks[0][: request.max_tokens]
    else:
        answer_text = "No relevant information found."

    return AnswerResponse(query=request.query, answer=answer_text, sources=sources)
