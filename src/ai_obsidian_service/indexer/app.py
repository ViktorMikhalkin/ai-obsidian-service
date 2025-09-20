from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from ai_obsidian_service.domain.models import Query

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
    """Get singleton indexer service instance."""
    return IndexerService()


app = FastAPI(
    title="AI Obsidian Service",
    description="Semantic search and RAG for Obsidian notes",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"ok": True, "errors": []}


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    service: IndexerService = Depends(get_indexer_service),
):
    """Search for similar content using semantic similarity."""
    try:
        # Create domain query object
        query = Query(text=request.query, top_k=request.top_k, filters=request.filters)

        # Perform search through service layer
        search_result = service.search_text(query.text, query.top_k)

        # Convert domain hits to API response
        api_hits = []
        for hit in search_result.hits:
            api_hit = SearchHit(
                id=f"{hit.doc_id}-{hit.chunk_order}",
                path=str(hit.doc_id),  # You might want to resolve actual file path
                kind="chunk",
                preview=hit.snippet,
                score=hit.score,
                start_char=hit.start_char,
                end_char=hit.end_char,
                metadata=hit.metadata,
            )
            api_hits.append(api_hit)

        return SearchResponse(
            results=api_hits,
            total_time_ms=search_result.total_time_ms,
            retrieved_at=search_result.retrieved_at,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}") from e


@app.post("/answer", response_model=AnswerResponse)
async def answer(
    request: AnswerRequest,
    service: IndexerService = Depends(get_indexer_service),
):
    """Generate answers based on retrieved context."""
    try:
        # Step 1: Perform semantic search
        query = Query(text=request.query, top_k=request.top_k)

        search_result = service.search_text(query.text, query.top_k)

        # Convert hits to API format for sources
        sources = []
        context_chunks = []

        for hit in search_result.hits:
            source = SearchHit(
                id=f"{hit.doc_id}-{hit.chunk_order}",
                path=str(hit.doc_id),
                kind="chunk",
                preview=hit.snippet,
                score=hit.score,
                start_char=hit.start_char,
                end_char=hit.end_char,
                metadata=hit.metadata,
            )
            sources.append(source)

            # Collect full text for context (not just snippet)
            # You might need to retrieve full chunk text from metadata
            context_chunks.append(hit.snippet)  # or full text if available

        # Step 2: Generate answer
        answer_text = ""
        if context_chunks:
            if (
                hasattr(service, "llm_client")
                and service.llm_client
                and hasattr(service.llm_client, "generate")
            ):
                # Use LLM for generative answer
                context = "\n\n".join(context_chunks)
                prompt = f"""Based on the following context, answer the question: {request.query}

Context:
{context}

Answer:"""
                answer_text = service.llm_client.generate(prompt)
            else:
                # Fallback to extractive answer
                answer_text = context_chunks[0][: request.max_tokens]
        else:
            answer_text = "No relevant information found."

        return AnswerResponse(
            query=request.query,
            answer=answer_text,
            sources=sources,
            total_time_ms=search_result.total_time_ms,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Answer generation error: {str(e)}"
        ) from e


@app.get("/index/stats")
async def get_index_stats(service: IndexerService = Depends(get_indexer_service)):
    """Get index statistics."""
    try:
        stats = service.get_stats()
        return {
            "total_chunks": stats.get("total_chunks", 0),
            "total_documents": stats.get("total_documents", 0),
            "index_size_mb": stats.get("index_size_mb", 0),
            "last_updated": stats.get("last_updated"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}") from e


@app.post("/index/rebuild")
async def rebuild_index(service: IndexerService = Depends(get_indexer_service)):
    """Rebuild the search index."""
    try:
        result = service.rebuild_index()
        return {"message": "Index rebuild completed", "stats": result}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Index rebuild error: {str(e)}"
        ) from e


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}
