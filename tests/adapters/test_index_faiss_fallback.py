import numpy as np

from ai_obsidian_service.adapters.index.faiss_index import FaissIndex
from ai_obsidian_service.core import Chunk, ChunkId, DocId, Query
from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery


def test_faiss_index_fallback_search():
    idx = FaissIndex(index_dir=None, dim=16)

    # Create chunks with proper id parameter
    chunks = [
        Chunk(
            id=ChunkId("d1_chunk_0"),
            doc_id=DocId("d1"),
            order=0,
            text="alpha beta gamma",
        ),
        Chunk(
            id=ChunkId("d2_chunk_0"),
            doc_id=DocId("d2"),
            order=0,
            text="delta epsilon zeta",
        ),
    ]

    # Convert to EmbeddedChunk objects with dummy embeddings
    embedded_chunks = [
        EmbeddedChunk(chunk=chunk, embedding=np.zeros(16, dtype=np.float32))
        for chunk in chunks
    ]

    idx.upsert(embedded_chunks)

    # Create EmbeddedQuery with dummy embedding
    query = Query(text="alpha", top_k=1)
    embedded_query = EmbeddedQuery(
        query=query, embedding=np.zeros(16, dtype=np.float32)
    )

    result = idx.search(embedded_query)

    # Access hits from the SearchResult
    assert result.hits and result.hits[0].doc_id == DocId("d1")
    assert 0.0 <= result.hits[0].score <= 1.0
