# tests/index/test_index_contract.py
from __future__ import annotations

from ai_obsidian_service.domain.models import Query


def test_empty_index_search_returns_empty(empty_search_service):
    """
    With an empty index, search should return no hits.
    Uses the special empty_search_service fixture (no indexing performed).
    """
    res = empty_search_service.search_text("anything", top_k=5)
    assert res.hits == []


def test_search_respects_topk_and_scores(search_service):
    """
    On the indexed mini_vault, ensure top_k is respected and scores are present.
    """
    q = Query("hello", top_k=1)
    res = search_service.search_text(q.text, top_k=q.top_k)
    assert len(res.hits) <= q.top_k
    for h in res.hits:
        assert isinstance(h.score, float)
