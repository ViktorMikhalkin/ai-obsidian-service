from __future__ import annotations

from typing import Any

from ai_obsidian_service.core import (
    Chunker,
    ChunkId,
    DocId,
    Document,
    DocumentParser,
    EmbeddingIndex,
    Hit,
    Query,
)
from ai_obsidian_service.domain.models import EmbeddedQuery, Chunk


def _embed_text(text: str, dim: int = 64):
    """Deterministic lightweight embedding (works with or without NumPy)."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        vec = [0.0] * dim
        if text:
            for i, ch in enumerate(text):
                vec[(ord(ch) + i) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm if norm > 0 else 0.0 for v in vec]
    else:
        v = np.zeros(dim, dtype=float)
        if text:
            for i, ch in enumerate(text):
                v[(ord(ch) + i) % dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


class SearchService:
    """Orchestrates parsing → chunking → indexing and search over EmbeddingIndex."""

    def __init__(
        self,
        parsers: list[DocumentParser],
        chunker: Chunker,
        index: EmbeddingIndex,
    ) -> None:
        self.parsers = parsers
        self.chunker = chunker
        self.index = index
        # meta by (doc_id, chunk_order)
        self._meta: dict[tuple[str, int], dict[str, Any]] = {}

# ---------- Indexing ----------

def index_document(self, doc: Document) -> int:
    """Index a parsed document and return number of chunks stored."""
    chunks: list[Chunk] = list(self.chunker.split(doc))
    self.index.upsert(chunks)
    return len(chunks)

def index_path(self, path: str) -> int:
    """Parse and index a single path; return number of chunks stored."""
    for p in self.parsers:
        if p.can_parse(path):
            doc: Document = p.parse(path)
            return self.index_document(doc)
    return 0

# ---------- Search ----------

def search_text(self, text: str, top_k: int = 5) -> list[Hit]:
    dim = getattr(self.index, "dim", 64)
    eq = EmbeddedQuery(
        query=Query(text=text, top_k=top_k), embedding=_embed_text(text, dim)
    )
    result = self.index.search(eq)
    return result.hits

# ---------- Resolve ----------

def resolve_meta(
    self, doc_id: DocId, chunk_id: ChunkId, order: int
) -> dict[str, Any]:
    key = (str(doc_id), int(order))
    return dict(self._meta.get(key, {}))

# ---------- Observability ----------

def get_stats(self) -> dict[str, object]:
    docs = {k[0] for k in self._meta.keys()}
    return {
        "total_documents": len(docs),
        "total_chunks": len(self._meta),
        "errors": [],
    }

# ---------- Lifecycle ----------

def shutdown(self) -> None:
    """Release resources gracefully (best-effort)."""
    idx = getattr(self, "index", None)
    for name in ("flush", "close", "shutdown"):
        fn = getattr(idx, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
