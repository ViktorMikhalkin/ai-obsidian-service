from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from ai_obsidian_service.domain.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    EmbeddedQuery,
    Query,
    SearchResult,
)


@runtime_checkable
class DocumentParser(Protocol):
    """Parses documents into domain objects."""

    def can_parse(self, path: str) -> bool: ...
    def parse(self, path: str) -> Document: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits documents into chunks."""

    def split(self, doc: Document) -> Sequence[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    """Generates vector embeddings from text."""

    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk: ...
    def embed_chunks(self, chunks: Iterable[Chunk]) -> Sequence[EmbeddedChunk]: ...
    def embed_query(self, query: Query) -> EmbeddedQuery: ...


@runtime_checkable
class VectorIndex(Protocol):
    """Low-level vector storage and search."""

    def add_vectors(self, vectors: np.ndarray, metadata: Sequence[dict]) -> None: ...
    def search_vectors(
        self, query_vector: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]: ...
    def save(self) -> None: ...
    def load(self) -> None: ...


@runtime_checkable
class EmbeddingIndex(Protocol):
    """High-level semantic search interface."""

    def upsert(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None: ...
    def search(self, embedded_query: EmbeddedQuery) -> SearchResult: ...


@runtime_checkable
class SearchService(Protocol):
    """Application service for end-to-end search."""

    def index_document(self, document: Document) -> None: ...
    def search_text(self, query_text: str, top_k: int = 5) -> SearchResult: ...


@runtime_checkable
class LlmClient(Protocol):
    """Language model client."""

    def generate(self, prompt: str) -> str: ...
