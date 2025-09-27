from __future__ import annotations

from pathlib import Path
from ai_obsidian_service.adapters.services.search_service import SearchService


def test_empty_index_search_returns_empty(search_service: SearchService):
    res = search_service.search_text("alpha", top_k=5)
    assert res.hits == []


def test_search_respects_topk_and_scores(search_service: SearchService, mini_vault: Path):
    # index markdown files
    for p in sorted(mini_vault.rglob("*.md")):
        search_service.index_path(str(p))

    res = search_service.search_text("charlie", top_k=2)
    assert 0 < len(res.hits) <= 2

    scores = [h.score for h in res.hits]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
