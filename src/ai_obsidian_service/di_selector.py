from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers import all_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api.endpoints.config import get_current_config
from ai_obsidian_service.core import Chunker  # Import the base protocol/interface
from ai_obsidian_service.index.embedder_sentence_transformers import (
    SentenceTransformersEmbedder,
)
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.index.memory_store import InMemoryVectorStore

if TYPE_CHECKING:
    pass

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
    index: Any
    search: SearchService


def _maybe_make_bm25() -> tuple[Any | None, int]:
    """Create BM25 reranker based on config."""
    config = get_current_config()

    if not config.bm25.enabled:
        return None, config.bm25.top_n
    if BM25RerankerType is None:
        log.debug("BM25Reranker module not available; rerank disabled.")
        return None, config.bm25.top_n
    try:
        return BM25RerankerType(), config.bm25.top_n
    except Exception as e:
        log.debug("BM25Reranker init failed: %s; rerank disabled.", e)
        return None, config.bm25.top_n


def _make_memory(*, model_name: str, chunker: Chunker) -> Components:
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
    *, model_name: str, index_dir: str | None, chunker: Chunker
) -> Components:
    from pathlib import Path

    config = get_current_config()
    embedder = SentenceTransformersEmbedder(model_name=model_name)

    # Always create empty store for fast startup
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

    # Use Any type to allow dynamic attribute assignment
    index: Any

    if config.indexing.enable_incremental:
        try:
            from ai_obsidian_service.index.enhanced_embedding_index import (
                EnhancedEmbeddingIndex,
            )

            # Default enable_dedup to True
            enable_dedup = True

            index = EnhancedEmbeddingIndex(
                embedder=embedder,
                store=store,
                chunker=chunker,
                enable_dedup=enable_dedup,
            )

            # Store index_dir for later background loading
            # Using Any type allows dynamic attributes
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
    )
    return Components(embedder=embedder, store=store, index=index, search=search)


def make_components(
    *, chunker: Chunker | None = None, index_dir: str | None = None
) -> Components:
    """
    Config-driven DI using configuration endpoint.
    Falls back to environment variables if config not available.
    """
    config = get_current_config()

    backend = config.indexing.backend or "memory"
    model_name = config.embeddings.model or "intfloat/multilingual-e5-small"

    # Use provided index_dir or fall back to config
    if index_dir is None:
        index_dir = config.indexing.index_dir

    # Create chunker if not provided
    actual_chunker: Chunker
    if chunker is None:
        # Chunker should already be created in container.py
        # This is a fallback
        if config.chunking.use_token_chunking:
            try:
                from ai_obsidian_service.adapters.chunkers.semantic_chunker import (
                    SemanticChunker,
                )

                actual_chunker = SemanticChunker(
                    max_tokens=config.chunking.target_tokens,
                    overlap_tokens=config.chunking.overlap_tokens,
                )
            except ImportError:
                actual_chunker = SimpleChunker(
                    max_chars=config.chunking.max_chars,
                    overlap=config.chunking.overlap_chars,
                )
        else:
            actual_chunker = SimpleChunker(
                max_chars=config.chunking.max_chars,
                overlap=config.chunking.overlap_chars,
            )
    else:
        actual_chunker = chunker

    if backend == "faiss":
        return _make_faiss(
            model_name=model_name, index_dir=index_dir, chunker=actual_chunker
        )
    return _make_memory(model_name=model_name, chunker=actual_chunker)
