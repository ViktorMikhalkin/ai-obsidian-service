from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers import default_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.di_selector import make_components as _base_make_components


@dataclass(slots=True)
class Components:
    """
    Unified DI bundle used by the app:
      - embedder / store / index are produced by the base selector (env-driven)
      - search is the SearchService bound to the chosen backend
      - parsers is the full, equal-footing set (MD + PDF + EPUB)
    """
    embedder: Any
    store: Any
    index: Any
    search: SearchService
    parsers: Sequence[Any]


def make_components(*, chunker: SimpleChunker | None = None) -> Components:
    """
    Build components using the existing env-driven selector, but
    attach ALL parsers (Markdown + PDF + EPUB) to SearchService.

    This keeps the selector logic (FAISS/memory, device, etc.) intact,
    and only augments the service with a multi-parser setup.
    """
    # 1) Build the base set (embedder, store, index, search) via the existing selector
    base = _base_make_components(chunker=chunker or SimpleChunker(max_chars=1000, overlap=100))

    # 2) Equal-footing parser set
    parsers = default_parsers()  # [MarkdownParser(), PdfParser(), EpubParser()]

    # 3) Inject parsers into the SearchService (supporting multiple shapes for backward-compat)
    search: SearchService = base.search  # type: ignore[assignment]
    if hasattr(search, "parsers"):
        # Newer API: explicit .parsers
        search.parsers = parsers  # type: ignore[attr-defined]
    elif hasattr(search, "set_parsers"):
        # Alternative API: setter
        search.set_parsers(parsers)  # type: ignore[attr-defined]
    else:
        # Fallback for very old single-parser API
        try:
            search.parser = parsers[0]  # type: ignore[attr-defined]
        except Exception:
            pass

    return Components(
        embedder=base.embedder,
        store=base.store,
        index=base.index,
        search=search,
        parsers=parsers,
    )
