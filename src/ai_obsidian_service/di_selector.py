from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers import all_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.index.embedder_sentence_transformers import (
    SentenceTransformersEmbedder,
)
from ai_obsidian_service.index.embedding_index import EmbeddingIndex

# NOTE: stores live directly under `index/`, not under `index/vector_store/`
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.index.memory_store import InMemoryVectorStore

# BM25 is optional: degrade gracefully if module is absent
try:
    from ai_obsidian_service.rerank.bm25 import BM25Reranker
    BM25RerankerType: type[Any] | None = BM25Reranker
except Exception:  # pragma: no cover
    BM25RerankerType = None

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Components:
    embedder: Any
    store: Any
    index: EmbeddingIndex
    search: SearchService


def _maybe_make_bm25() -> tuple[Any | None, int]:
    """
    ENABLE_BM25 = "1" | "0" (default "1")
    BM25_TOPN   = int       (default 50)
    """
    enabled = (os.getenv("ENABLE_BM25", "1") == "1")
    topn = int(os.getenv("BM25_TOPN", "50"))
    if not enabled:
        return None, topn
    if BM25RerankerType is None:
        log.debug("BM25Reranker module not available; rerank disabled.")
        return None, topn
    try:
        return BM25RerankerType(), topn
    except Exception as e:  # pragma: no cover
        log.debug("BM25Reranker init failed: %s; rerank disabled.", e)
        return None, topn


def _make_memory(*, model_name: str, chunker: SimpleChunker) -> Components:
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    store = InMemoryVectorStore(dim=None)
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)

    parsers = all_parsers()  # MD + PDF + EPUB
    reranker, rerank_topn = _maybe_make_bm25()
    search = SearchService(index=index, parsers=parsers, reranker=reranker, rerank_topn=rerank_topn)
    return Components(embedder=embedder, store=store, index=index, search=search)


def _make_faiss(*, model_name: str, index_dir: str | None, chunker: SimpleChunker) -> Components:
    from pathlib import Path

    embedder = SentenceTransformersEmbedder(model_name=model_name)

    # Try to load existing index from disk
    store = None
    if index_dir and Path(index_dir).exists() and (Path(index_dir) / "meta.json").exists():
        try:
            log.info(f"Loading existing FAISS index from {index_dir}")
            store = FaissVectorStore.load(index_dir, expected_model_name=model_name)
            log.info(f"Loaded index with {store.count} chunks")
        except Exception as e:
            log.warning(f"Failed to load index from {index_dir}: {e}. Creating new empty store.")
            store = None

    if store is None:
        log.info("Creating new empty FAISS store")
        store = FaissVectorStore(dim=None)

    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)

    parsers = all_parsers()  # MD + PDF + EPUB
    reranker, rerank_topn = _maybe_make_bm25()
    search = SearchService(index=index, parsers=parsers, reranker=reranker, rerank_topn=rerank_topn)
    return Components(embedder=embedder, store=store, index=index, search=search)


def make_components(*, chunker: SimpleChunker | None = None) -> Components:
    """
    Env-driven DI:
      VECTOR_STORE_BACKEND = memory | faiss
      EMBEDDINGS_MODEL     = sentence-transformers model (default: all-MiniLM-L6-v2)
      INDEX_DIR            = path for FAISS persistence (faiss backend)
      ENABLE_BM25          = 1|0 (default 1)
      BM25_TOPN            = int (default 50)
    """
    backend = (os.getenv("VECTOR_STORE_BACKEND") or "memory").lower()
    model_name = os.getenv("EMBEDDINGS_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
    index_dir = os.getenv("INDEX_DIR")
    chunker = chunker or SimpleChunker(max_chars=1500, overlap=150)

    if backend == "faiss":
        return _make_faiss(model_name=model_name, index_dir=index_dir, chunker=chunker)
    return _make_memory(model_name=model_name, chunker=chunker)
