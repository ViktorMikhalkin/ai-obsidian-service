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
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.index.memory_store import InMemoryVectorStore

try:
    from ai_obsidian_service.rerank.bm25 import BM25Reranker

    BM25RerankerType: type[Any] | None = BM25Reranker
except Exception:
    BM25RerankerType = None

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Components:
    embedder: Any
    store: Any
    index: Any  # Can be EmbeddingIndex or EnhancedEmbeddingIndex
    search: SearchService


def _maybe_make_bm25() -> tuple[Any | None, int]:
    """
    ENABLE_BM25 = "1" | "0" (default "1")
    BM25_TOPN   = int       (default 50)
    """
    enabled = os.getenv("ENABLE_BM25", "1") == "1"
    topn = int(os.getenv("BM25_TOPN", "50"))
    if not enabled:
        return None, topn
    if BM25RerankerType is None:
        log.debug("BM25Reranker module not available; rerank disabled.")
        return None, topn
    try:
        return BM25RerankerType(), topn
    except Exception as e:
        log.debug("BM25Reranker init failed: %s; rerank disabled.", e)
        return None, topn


def _get_chunker() -> SimpleChunker:
    """
    Get chunker based on environment settings.

    USE_TOKEN_CHUNKING=1 -> SemanticChunker (token-based)
    USE_TOKEN_CHUNKING=0 -> SimpleChunker (char-based, default)
    """
    use_token_chunking = os.getenv("USE_TOKEN_CHUNKING", "0") == "1"

    if use_token_chunking:
        try:
            from ai_obsidian_service.adapters.chunkers.semantic_chunker import (
                SemanticChunker,
            )

            max_tokens = int(os.getenv("CHUNK_MAX_TOKENS", "750"))
            overlap_tokens = int(os.getenv("CHUNK_OVERLAP_TOKENS", "75"))
            log.info(
                f"Using SemanticChunker: max_tokens={max_tokens}, overlap={overlap_tokens}"
            )
            return SemanticChunker(max_tokens=max_tokens, overlap_tokens=overlap_tokens)  # type: ignore
        except ImportError:
            log.warning("SemanticChunker not available, falling back to SimpleChunker")

    max_chars = int(os.getenv("CHUNK_MAX_CHARS", "1500"))
    overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "150"))
    log.info(f"Using SimpleChunker: max_chars={max_chars}, overlap={overlap_chars}")
    return SimpleChunker(max_chars=max_chars, overlap=overlap_chars)


def _make_memory(*, model_name: str, chunker: SimpleChunker) -> Components:
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    store = InMemoryVectorStore(dim=None)
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)

    parsers = all_parsers()
    reranker, rerank_topn = _maybe_make_bm25()
    search = SearchService(
        index=index, parsers=parsers, reranker=reranker, rerank_topn=rerank_topn
    )
    return Components(embedder=embedder, store=store, index=index, search=search)


def _make_faiss(
    *, model_name: str, index_dir: str | None, chunker: SimpleChunker
) -> Components:
    from pathlib import Path

    embedder = SentenceTransformersEmbedder(model_name=model_name)

    # Always create empty store for fast startup
    # Loading happens asynchronously in the lifespan
    store = FaissVectorStore(dim=None)
    log.info("FAISS store initialized (empty)")

    # Check if we should load from disk later
    should_load = (
        index_dir
        and Path(index_dir).exists()
        and (Path(index_dir) / "meta.json").exists()
    )

    if should_load:
        log.info(f"Will load existing index from {index_dir} in background")
    else:
        log.info("No existing index found - starting fresh")

    # Check if incremental indexing is enabled
    use_incremental = os.getenv("ENABLE_INCREMENTAL_INDEXING", "1") == "1"

    # Type annotation to help MyPy
    index: EmbeddingIndex | Any

    if use_incremental:
        try:
            from ai_obsidian_service.index.enhanced_embedding_index import (
                EnhancedEmbeddingIndex,
            )

            enable_dedup = os.getenv("ENABLE_CHUNK_DEDUP", "1") == "1"

            index = EnhancedEmbeddingIndex(
                embedder=embedder,
                store=store,
                chunker=chunker,
                enable_dedup=enable_dedup,
            )

            # Store index_dir for later background loading
            index._pending_load_path = index_dir if should_load else None
            index._pending_load_model = model_name if should_load else None

            log.info("Using EnhancedEmbeddingIndex with incremental indexing")

        except ImportError:
            log.warning(
                "EnhancedEmbeddingIndex not available, falling back to basic EmbeddingIndex"
            )
            index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
            index._pending_load_path = index_dir if should_load else None
            index._pending_load_model = model_name if should_load else None
    else:
        # Use basic EmbeddingIndex
        index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
        index._pending_load_path = index_dir if should_load else None
        index._pending_load_model = model_name if should_load else None
        log.info("Using basic EmbeddingIndex (incremental disabled)")

    parsers = all_parsers()
    reranker, rerank_topn = _maybe_make_bm25()
    search = SearchService(
        index=index, parsers=parsers, reranker=reranker, rerank_topn=rerank_topn
    )  # type: ignore[arg-type]
    return Components(embedder=embedder, store=store, index=index, search=search)


def make_components(*, chunker: SimpleChunker | None = None) -> Components:
    """
    Env-driven DI:
      VECTOR_STORE_BACKEND = memory | faiss (default: memory)
      EMBEDDINGS_MODEL     = sentence-transformers model (default: multilingual-e5-small)
      INDEX_DIR            = path for FAISS persistence (faiss backend)
      ENABLE_BM25          = 1|0 (default 1)
      BM25_TOPN            = int (default 50)

      Optimization flags:
      USE_TOKEN_CHUNKING          = 1|0 (default 0) - Use SemanticChunker with tiktoken
      ENABLE_INCREMENTAL_INDEXING = 1|0 (default 1) - Skip unchanged documents
      ENABLE_CHUNK_DEDUP          = 1|0 (default 1) - Enable chunk deduplication
      CHUNK_MAX_TOKENS            = int (default 750) - For token-based chunking
      CHUNK_OVERLAP_TOKENS        = int (default 75) - For token-based chunking
      CHUNK_MAX_CHARS             = int (default 1500) - For char-based chunking
      CHUNK_OVERLAP_CHARS         = int (default 150) - For char-based chunking
    """
    backend = (os.getenv("VECTOR_STORE_BACKEND") or "memory").lower()
    model_name = os.getenv("EMBEDDINGS_MODEL") or "intfloat/multilingual-e5-small"
    index_dir = os.getenv("INDEX_DIR")

    if chunker is None:
        chunker = _get_chunker()

    if backend == "faiss":
        return _make_faiss(model_name=model_name, index_dir=index_dir, chunker=chunker)
    return _make_memory(model_name=model_name, chunker=chunker)
