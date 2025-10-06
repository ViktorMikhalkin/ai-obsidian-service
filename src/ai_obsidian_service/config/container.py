from __future__ import annotations

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers import all_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.api.endpoints.config import get_current_config
from ai_obsidian_service.core import Chunker
from ai_obsidian_service.di_selector import make_components
from ai_obsidian_service.usecases.index_corpus import IndexCorpus


def build_search_service(index_dir: str | None = None) -> SearchService:
    """
    Build a production SearchService using configuration endpoint.
    Falls back to provided index_dir if config not available.
    """
    config = get_current_config()

    parsers = all_parsers()

    # Use config for chunking strategy
    chunker: Chunker
    if config.chunking.use_token_chunking:
        try:
            from ai_obsidian_service.adapters.chunkers.semantic_chunker import (
                SemanticChunker,
            )

            chunker = SemanticChunker(
                max_tokens=config.chunking.target_tokens,
                overlap_tokens=config.chunking.overlap_tokens,
            )
        except ImportError:
            # Fall back to SimpleChunker if SemanticChunker not available
            chunker = SimpleChunker(
                max_chars=config.chunking.max_chars,
                overlap=config.chunking.overlap_chars,
            )
    else:
        chunker = SimpleChunker(
            max_chars=config.chunking.max_chars, overlap=config.chunking.overlap_chars
        )

    # Use config index_dir if not provided as parameter
    if index_dir is None:
        index_dir = config.indexing.index_dir

    di = make_components(chunker=chunker, index_dir=index_dir)
    service: SearchService = di.search

    # Ensure the service knows about *all* parsers
    if hasattr(service, "parsers"):
        service.parsers = parsers
    elif hasattr(service, "set_parsers"):
        service.set_parsers(parsers)
    else:
        try:
            service.parser = parsers[0]  # type: ignore[attr-defined]
        except Exception:
            pass

    return service


def build_index_corpus(index_dir: str | None = None) -> IndexCorpus:
    """
    Build the use-case for bulk indexing a directory.
    Uses configuration endpoint for settings.
    """
    config = get_current_config()

    parsers = all_parsers()

    # Use config for chunking strategy
    chunker: Chunker
    if config.chunking.use_token_chunking:
        try:
            from ai_obsidian_service.adapters.chunkers.semantic_chunker import (
                SemanticChunker,
            )

            chunker = SemanticChunker(
                max_tokens=config.chunking.target_tokens,
                overlap_tokens=config.chunking.overlap_tokens,
            )
        except ImportError:
            chunker = SimpleChunker(
                max_chars=config.chunking.max_chars,
                overlap=config.chunking.overlap_chars,
            )
    else:
        chunker = SimpleChunker(
            max_chars=config.chunking.max_chars, overlap=config.chunking.overlap_chars
        )

    # Use config index_dir if not provided
    if index_dir is None:
        index_dir = config.indexing.index_dir

    service = build_search_service(index_dir=index_dir)
    return IndexCorpus(parsers=parsers, chunker=chunker, service=service)
