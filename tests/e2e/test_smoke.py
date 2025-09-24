
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.parsers import default_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import DocId, Document
from ai_obsidian_service.domain.models import EmbeddedChunk, Hit, SearchResult
from ai_obsidian_service.index.embedding_index import EmbeddingIndex


# ---- in-memory fake embedder/store to keep e2e simple and fast ----
class FakeEmbedder:
    dim = 4
    def embed(self, text: str) -> NDArray[np.float32]:
        v = np.zeros(self.dim, dtype=np.float32)
        v[:] = len(text) % 7
        return v

class MemoryStore:
    def __init__(self) -> None:
        self.rows: list[EmbeddedChunk] = []

    def upsert(self, embedded: list[EmbeddedChunk]) -> None:
        self.rows.extend(embedded)

    def search(self, q_vec: NDArray[np.float32], top_k: int) -> SearchResult:
        qv = float(q_vec[0])
        scored: list[tuple[float, EmbeddedChunk]] = []
        for e in self.rows:
            s = -abs(len(e.chunk.text) % 7 - qv)
            scored.append((s, e))
        scored.sort(key=lambda t: t[0], reverse=True)
        hits = [Hit(chunk=e.chunk, score=float(s)) for s, e in scored[:top_k]]
        return SearchResult(query=None, hits=hits)  # type: ignore[arg-type]

@pytest.mark.e2e
def test_mini_corpus_and_collection_filter(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "notes").mkdir(parents=True)
    (vault / "docs").mkdir(parents=True)

    (vault / "notes" / "A.md").write_text("# A\napple banana\n", encoding="utf-8")
    (vault / "notes" / "B.md").write_text("# B\nbanana cherry\n", encoding="utf-8")
    (vault / "docs" / "X.md").write_text("# X\nzebra kiwi\n", encoding="utf-8")

    parsers = default_parsers()
    mdp = parsers[0]  # MarkdownParser
    chunker = SimpleChunker(max_chars=1000, overlap=100)

    embedder = FakeEmbedder()
    store = MemoryStore()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker, state_dir=str(tmp_path / ".state"))
    svc = SearchService(index=index, parser=mdp)

    # index notes/
    for p in (vault / "notes").glob("*.md"):
        doc = mdp.parse(str(p))
        rel = p.relative_to(vault).as_posix()
        meta = dict(doc.metadata or {})
        coll = "/".join(rel.split("/")[:-1]) if "/" in rel else ""
        meta.update({"path": rel, "collection": coll})
        doc = Document(id=DocId(rel), path=str(p), mime=doc.mime, text=doc.text, metadata=meta)
        svc.index_document(doc)

    # index docs/
    for p in (vault / "docs").glob("*.md"):
        doc = mdp.parse(str(p))
        rel = p.relative_to(vault).as_posix()
        meta = dict(doc.metadata or {})
        coll = "/".join(rel.split("/")[:-1]) if "/" in rel else ""
        meta.update({"path": rel, "collection": coll})
        doc = Document(id=DocId(rel), path=str(p), mime=doc.mime, text=doc.text, metadata=meta)
        svc.index_document(doc)

    result = svc.search_text("banana", top_k=5, collection="notes")
    assert len(result.hits) >= 1
    assert all(h.chunk.metadata.get("collection") == "notes" for h in result.hits)
