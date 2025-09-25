from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

# Using simple str - this eliminates a lot of mypy errors in tests
DocId = str
ChunkId = str


@dataclass(slots=True)
class Document:
    id: DocId
    path: str
    mime: str
    text: str
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class Chunk:
    id: ChunkId
    doc_id: DocId
    order: int
    text: str
    # Compatibility: support both meta and metadata
    metadata: dict[str, Any] | None = None
    meta: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None and self.meta is not None:
            self.metadata = self.meta
        elif self.meta is None and self.metadata is not None:
            self.meta = self.metadata


@dataclass(slots=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: np.ndarray  # 1D float32


# For legacy FAISS adapter
@dataclass(slots=True)
class EmbeddedQuery:
    text: str | None
    vector: np.ndarray


@dataclass(slots=True)
class Query:
    text: str
    top_k: int = 5


@dataclass(slots=True)
class Hit:
    # modern path - through chunk
    chunk: Chunk | None = None
    score: float = 0.0
    # legacy fields (so old calls with named arguments don't fail)
    doc_id: DocId | None = None
    chunk_id: ChunkId | None = None
    chunk_order: int | None = None
    snippet: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.chunk is not None:
            self.doc_id = self.doc_id or self.chunk.doc_id
            self.chunk_id = self.chunk_id or self.chunk.id
            self.chunk_order = self.chunk_order if self.chunk_order is not None else self.chunk.order
            if self.metadata is None:
                self.metadata = self.chunk.metadata


@dataclass(slots=True)
class SearchResult:
    query: Query | None
    hits: list[Hit] = field(default_factory=list)
    # for convenient metrics (assigned after construction)
    total_time_ms: float | None = None
    retrieved_at: datetime | None = None
