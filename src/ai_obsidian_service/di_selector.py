from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.di import DummyEmbedder, InMemoryVectorStore
from ai_obsidian_service.domain.models import EmbeddedChunk, SearchResult
from ai_obsidian_service.index.embedder import Embedder
from ai_obsidian_service.index.embedder_sentence_transformers import (
    SentenceTransformersEmbedder,
)
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.index.vector_store import (
    VectorStore,  # <-- index-layer protocol
)
from ai_obsidian_service.ports.interfaces import (
    Chunker,  # canonical protocol from ports
)
from ai_obsidian_service.rerank.bm25 import BM25Reranker

__all__ = ["make_components", "Components", "Chunker"]


@dataclass(slots=True)
class Components:
    embedder: Embedder
    store: VectorStore
    index: EmbeddingIndex
    parser: MarkdownParser
    search: SearchService


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() not in ("0", "false", "no")


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    try:
        return int(val) if val is not None else default
    except Exception:
        return default


class _StoreAdapter(VectorStore):
    """
    Adapter to widen InMemoryVectorStore.upsert from list[...] to Sequence[...],
    so it satisfies the VectorStore protocol expected by EmbeddingIndex.
    """
    def __init__(self, inner: InMemoryVectorStore) -> None:
        self._inner = inner
        # propagate common attrs (optional)
        self.dim = getattr(inner, "dim", 384)

    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        self._inner.upsert(list(chunks))

    def search(self, query_vec: np.ndarray, top_k: int) -> SearchResult:
        return self._inner.search(query_vec, top_k)


def _make_memory(*, chunker: Chunker) -> Components:
    embedder: Embedder = DummyEmbedder()
    store: VectorStore = _StoreAdapter(InMemoryVectorStore())
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)

    enable_bm25 = _env_bool("AIOS_BM25", True)
    rerank_topn = _env_int("AIOS_RERANK_TOPN", 50)
    search = SearchService(
        index=index,
        parser=parser,
        reranker=(BM25Reranker() if enable_bm25 else None),
        rerank_topn=rerank_topn,
    )
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)


def _make_faiss(*, chunker: Chunker, model_name: str) -> Components:
    embedder: Embedder = SentenceTransformersEmbedder(model_name=model_name)
    store: VectorStore = FaissVectorStore()
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)

    enable_bm25 = _env_bool("AIOS_BM25", True)
    rerank_topn = _env_int("AIOS_RERANK_TOPN", 50)
    search = SearchService(
        index=index,
        parser=parser,
        reranker=(BM25Reranker() if enable_bm25 else None),
        rerank_topn=rerank_topn,
    )
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)


def make_components(
        *,
        chunker: Chunker,
        backend: str | None = None,
        model_name: str | None = None,
) -> Components:
    """
    Build iteration-5 components.

    - backend: "memory" | "faiss" (defaults to env VECTOR_STORE_BACKEND or "memory")
    - model_name: ST model when backend="faiss"
    """
    _backend = (backend or os.getenv("VECTOR_STORE_BACKEND") or "memory").strip().lower()
    if _backend == "memory":
        return _make_memory(chunker=chunker)
    if _backend == "faiss":
        _model = model_name or os.getenv("ST_MODEL_NAME") or "sentence-transformers/all-MiniLM-L6-v2"
        return _make_faiss(chunker=chunker, model_name=_model)
    raise ValueError(f"Unsupported VECTOR_STORE_BACKEND={_backend!r}")
