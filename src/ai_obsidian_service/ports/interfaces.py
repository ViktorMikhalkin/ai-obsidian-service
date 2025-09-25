from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ai_obsidian_service.domain.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    EmbeddedQuery,
    SearchResult,
)


@runtime_checkable
class DocumentParser(Protocol):
    def can_parse(self, path: str) -> bool: ...
    def parse(self, path: str) -> Document: ...
    # parse_text остаётся в реализациях; из протокола убираем

@runtime_checkable
class Chunker(Protocol):
    def split(self, doc: Document) -> Sequence[Chunk]: ...

@runtime_checkable
class Embedder(Protocol):
    def embed(self, text: str) -> np.ndarray: ...

@runtime_checkable
class VectorStore(Protocol):
    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None: ...
    def search(self, query_vec: np.ndarray, top_k: int) -> SearchResult: ...

@runtime_checkable
class EmbeddingIndex(Protocol):
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...
    def index_document(self, doc: Document) -> int: ...
    def search(self, embedded_query: EmbeddedQuery | str, top_k: int = 5) -> SearchResult: ...

@runtime_checkable
class SearchService(Protocol):
    def index_document(self, document: Document) -> int: ...
    def search_text(self, query_text: str, top_k: int = 5, collection: str | None = None) -> SearchResult: ...
    def resolve_meta(self, chunk_id: str) -> dict[str, Any]: ...
    def shutdown(self) -> None: ...

@runtime_checkable
class LlmClient(Protocol):
    def generate(self, prompt: str) -> str: ...
