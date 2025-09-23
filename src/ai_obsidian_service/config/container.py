from __future__ import annotations

from ai_obsidian_service.adapters.parsers import default_parsers
from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.index.faiss_index import FaissIndex
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.usecases.index_corpus import IndexCorpus


def build_search_service(index_dir: str | None = None) -> SearchService:
    """
    Build a "proper" service:
      - parsers: default_parsers()
      - chunker: SimpleChunker(max_chars=1000, overlap=100)
      - index  : FaissIndex(index_dir, dim=64)  # index adapter implementing .upsert(chunks), .search(Query)
    """
    parsers = default_parsers()
    chunker = SimpleChunker(max_chars=1000, overlap=100)
    index = FaissIndex(index_dir=index_dir, dim=64)
    return SearchService(parsers=parsers, chunker=chunker, index=index)


def build_index_corpus(index_dir: str | None = None) -> IndexCorpus:
    """
    Build use-case for bulk indexing of a directory.
    Delegates indexing to SearchService.
    """
    parsers = default_parsers()
    chunker = SimpleChunker(max_chars=1000, overlap=100)
    service = build_search_service(index_dir=index_dir)
    return IndexCorpus(parsers=parsers, chunker=chunker, service=service)