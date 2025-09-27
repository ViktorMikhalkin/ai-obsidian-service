from __future__ import annotations

from pathlib import Path
from ai_obsidian_service.adapters.services.search_service import SearchService


def test_collection_filter_works(search_service: SearchService, mini_vault: Path):
    for p in sorted(mini_vault.rglob("*.md")):
        search_service.index_path(str(p))

    res = search_service.search_text("charlie", top_k=10, collection="notes")
    assert res.hits
    for h in res.hits:
        meta = (h.metadata or (h.chunk.metadata if h.chunk else None)) or {}
        assert meta.get("collection") == "notes"
