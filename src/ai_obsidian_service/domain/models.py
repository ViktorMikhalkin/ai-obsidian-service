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
    doc_id: DocId
    chunk_id: ChunkId
    chunk_order: int
    score: float
    snippet: str
    chunk: Chunk | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # если metadata не задана — аккуратно достать из chunk, если она там есть
        if self.metadata is None:
            chunk_meta = getattr(self.chunk, "metadata", None) if self.chunk is not None else None
            self.metadata = chunk_meta or {}


@dataclass(slots=True)
class SearchResult:
    query: Query | None
    hits: list[Hit] = field(default_factory=list)
    # for convenient metrics (assigned after construction)
    total_time_ms: float | None = None
    retrieved_at: datetime | None = None
