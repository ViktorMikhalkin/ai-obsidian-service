
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
    metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    chunk: Chunk
    embedding: np.ndarray  # 1D float32


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    top_k: int = 5


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class SearchResult:
    query: Query | None
    hits: list[Hit]
