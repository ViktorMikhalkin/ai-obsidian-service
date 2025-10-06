"""
Public façade for domain entities and ports.
"""

from .domain.models import Chunk, ChunkId, DocId, Document, Hit, Query, SearchResult
from .ports.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingIndex,
    LlmClient,
    SearchService,
)

__all__ = [
    # Domain
    "DocId",
    "ChunkId",
    "Document",
    "Chunk",
    "Query",
    "Hit",
    "SearchResult",
    # Ports
    "DocumentParser",
    "Chunker",
    "EmbeddingIndex",
    "LlmClient",
    "SearchService",
]
