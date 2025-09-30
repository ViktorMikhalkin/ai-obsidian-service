from __future__ import annotations

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers import all_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.di_selector import make_components
from ai_obsidian_service.usecases.index_corpus import IndexCorpus


def build_search_service(index_dir: str | None = None) -> SearchService:
    """
    Build a production SearchService:
      - parsers: Markdown + PDF + EPUB (equal footing)
      - chunker: SimpleChunker(max_chars=1500, overlap=150)
      - embedder/store/index: selected in di_selector.make_components()
        (backends decided via env, e.g. VECTOR_STORE_BACKEND)
    Note:
      - index_dir may be used by the underlying store if it persists to disk.
      - we attach the parsers set directly to the returned service.
    """
    parsers = all_parsers()
    chunker = SimpleChunker(max_chars=1500, overlap=150)
    di = make_components(chunker=chunker)
    service: SearchService = di.search  # returned by DI factory

    # Ensure the service knows about *all* parsers
    if hasattr(service, "parsers"):
        # Some versions expose `.parsers` explicitly
        service.parsers = parsers
    elif hasattr(service, "set_parsers"):
        # Or a setter is available
        service.set_parsers(parsers)
    else:
        # As a last resort, keep compatibility with older signatures that accepted a single parser
        # by setting a primary parser and letting the service select internally if it supports it.
        try:
            service.parser = parsers[0]  # type: ignore[attr-defined]
        except Exception:
            pass

    return service


def build_index_corpus(index_dir: str | None = None) -> IndexCorpus:
    """
    Build the use-case for bulk indexing a directory. It delegates actual parsing/chunking/upsert
    to the SearchService built above.
    """
    parsers = all_parsers()
    chunker = SimpleChunker(max_chars=1500, overlap=150)
    service = build_search_service(index_dir=index_dir)
    return IndexCorpus(parsers=parsers, chunker=chunker, service=service)