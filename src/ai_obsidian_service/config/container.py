from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.di_selector import make_components
from ai_obsidian_service.ports.interfaces import Chunker as ChunkerPort
from ai_obsidian_service.usecases.index_corpus import IndexCorpus

chunker: ChunkerPort = SimpleChunker(max_chars=1000, overlap=100)
cmp = make_components(chunker=chunker)


def build_search_service(index_dir: str | None = None) -> SearchService:
    """
    Build an iteration-5 SearchService:
      - parser  : MarkdownParser (concrete)
      - chunker : SimpleChunker(max_chars=1000, overlap=100)
      - embedder/store/index come from di_selector.make_components(...)
        (backend is chosen via env: VECTOR_STORE_BACKEND=memory|faiss)

    If a store supports persistence and exposes `index_dir`, we set it when provided.
    """
    parser = MarkdownParser()
    chunker: ChunkerPort = SimpleChunker(max_chars=1000, overlap=100)

    # Build core components (embedder + vector store + index + search service)
    cmp = make_components(chunker=chunker)
    service: SearchService = cmp.search

    # Ensure the service uses our concrete parser explicitly
    service.parser = parser  # the service expects a DocumentParser

    # Optional: forward index_dir to the underlying store if it supports it
    if index_dir is not None:
        store: Any = getattr(cmp.index, "store", None)
        if store is not None and hasattr(store, "index_dir"):
            store.index_dir = index_dir
            try:
                Path(index_dir).mkdir(parents=True, exist_ok=True)
            except PermissionError as e:  # keep a friendly error message
                raise RuntimeError(f"Cannot create index_dir {index_dir}: {e}") from e

    return service


def build_index_corpus(index_dir: str | None = None) -> IndexCorpus:
    """
    Build the use case for bulk indexing of a directory, delegating to SearchService.
    We pass a concrete parser list and the same chunker, to keep the pipeline explicit.
    """
    parser = MarkdownParser()
    chunker: ChunkerPort = SimpleChunker(max_chars=1000, overlap=100)
    service = build_search_service(index_dir=index_dir)
    # IndexCorpus expects parsers (sequence), chunker, and the service
    return IndexCorpus(parsers=[parser], chunker=chunker, service=service)
