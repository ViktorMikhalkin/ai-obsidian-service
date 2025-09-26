from __future__ import annotations

from datetime import datetime

import pytest

from ai_obsidian_service.core import ChunkId, DocId, Hit, Query, SearchResult
from ai_obsidian_service.rag import answer_with_citations


class _Chunk:
    def __init__(self, text: str, path: str = "docs/x.md") -> None:
        self.text = text
        self.meta = {"path": path}


def _sr(snippet: str, text: str = "lorem ipsum dolor") -> SearchResult:
    q = Query(text="q", top_k=5)
    ch = _Chunk(text=text, path="notes/B.md")
    hit = Hit(
        doc_id=DocId("d"),
        chunk_id=ChunkId("d:0"),
        chunk_order=0,
        score=0.5,
        snippet=snippet,
        chunk=ch,  # type: ignore[arg-type]
    )
    return SearchResult(query=q, hits=[hit], total_time_ms=0.1, retrieved_at=datetime.now())


def test_answer_with_citations_mini_mode() -> None:
    res = _sr(snippet="ipsum dolor", text="lorem ipsum dolor sit amet")
    answer, cites = answer_with_citations("q", res, llm=None)
    assert isinstance(answer, str) and len(answer) > 0
    assert len(cites) == 1
    c0 = cites[0]
    assert c0["doc_path"] == "notes/B.md"
    # span points into original chunk text
    assert c0["span"][0] >= -1 and c0["span"][1] >= -1


def test_answer_with_citations_llm_path() -> None:
    class _LLM:
        def generate(self, prompt: str, system: str | None = None) -> str:
            assert "Context:" in prompt
            return "LLM ok"

    res = _sr(snippet="hello world", text="hello world and others")
    answer, cites = answer_with_citations("q", res, llm=_LLM())
    assert answer == "LLM ok"
    assert len(cites) == 1
