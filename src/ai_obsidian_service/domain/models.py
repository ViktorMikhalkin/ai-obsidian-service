from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

import numpy as np

DocId = NewType("DocId", str)
ChunkId = NewType("ChunkId", str)


@dataclass(frozen=True, slots=True)
class Document:
    id: DocId
    path: str
    mime: str
    text: str
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class Chunk:
    id: ChunkId
    doc_id: DocId
    order: int
    text: str
    start_char: int = 0
    end_char: int | None = None
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Chunk with its vector embedding - used by infrastructure layer."""

    chunk: Chunk
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    top_k: int = 5
    filters: dict | None = None


@dataclass(frozen=True, slots=True)
class EmbeddedQuery:
    """Query with its vector embedding - used by infrastructure layer."""

    query: Query
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class Hit:
    chunk_id: ChunkId
    doc_id: DocId
    chunk_order: int
    score: float
    snippet: str
    start_char: int = 0
    end_char: int | None = None
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Container for search results with metadata."""

    query: Query
    hits: list[Hit]
    total_time_ms: float
    retrieved_at: str
