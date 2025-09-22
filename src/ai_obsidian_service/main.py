from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException

from .indexer.schemas import (
    AnswerRequest,
    AnswerResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
# Fix: Import the correct service class name
from .indexer.services import IndexerService  # Adjust this import based on actual class name


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
        # Use the new enhanced search method
        search_result = service.search_text(request.query, request.top_k)

        # Convert domain hits to API response
        api_hits = []
        for hit in search_result.hits:
            # Fix: Only use fields that exist in SearchHit schema
            api_hit = SearchHit(
                id=f"{hit.doc_id}-{hit.chunk_order}",
                path=str(hit.doc_id),
                kind="chunk",
                preview=hit.snippet,
                score=hit.score,
                # Remove: start_char, end_char, metadata if not in schema
            )
            api_hits.append(api_hit)

        # Fix: Only use fields that exist in SearchResponse schema
        return SearchResponse(
            results=api_hits,
            # Remove: total_time_ms, retrieved_at if not in schema
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
        search_result = service.search_text(request.query, request.top_k)

        # Convert hits to API format for sources
        sources = []
        context_chunks = []

        for hit in search_result.hits:
            # Fix: Only use fields that exist in SearchHit schema
            source = SearchHit(
                id=f"{hit.doc_id}-{hit.chunk_order}",
                path=str(hit.doc_id),
                kind="chunk",
                preview=hit.snippet,
                score=hit.score,
                # Remove: start_char, end_char, metadata if not in schema
            )
            sources.append(source)

            # Collect full text for context (use snippet for now)
            context_chunks.append(hit.snippet)

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
                # Fix: Remove reference to max_tokens if it doesn't exist on AnswerRequest
                # Fallback to extractive answer with fixed length
                answer_text = context_chunks[0][:500]  # Use fixed limit instead
        else:
            answer_text = "No relevant information found."

        # Fix: Only use fields that exist in AnswerResponse schema
        return AnswerResponse(
            query=request.query,
            answer=answer_text,
            sources=sources,
            # Remove: total_time_ms if not in schema
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
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}") from e


@app.post("/index/rebuild")
async def rebuild_index(service: IndexerService = Depends(get_indexer_service)):
    """Rebuild the search index."""
    try:
        result = service.rebuild_index()
        return result
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