"""
Public façade for domain entities and ports (GRASP/SOLID layer).
Safe to import in type hints, contracts, and application assembly.
This module has no heavy side-effects and imports only lightweight symbols.
"""

from .domain.models import Chunk, DocId, Document, Hit, Query
from .ports.interfaces import Chunker, DocumentParser, EmbeddingIndex, LlmClient

__all__ = [
    "DocId",
    "Document",
    "Chunk",
    "Query",
    "Hit",
    "DocumentParser",
    "Chunker",
    "EmbeddingIndex",
    "LlmClient",
]
