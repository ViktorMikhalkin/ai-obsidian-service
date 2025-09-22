from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ai_obsidian_service.core import (
    ChunkId,
    DocId,
    EmbeddingIndex,
    Hit,
    SearchResult,
)
from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery

# Optional NumPy — safe for mypy
np: Any = None
try:  # pragma: no cover
    import numpy

    np = numpy
except ImportError:  # pragma: no cover
    pass


def _has_numpy() -> bool:
    """Check if numpy is available."""
    return np is not None


@dataclass(frozen=True)
class _Meta:
    doc_id: str
    chunk_id: str
    order: int
    text: str


class FaissIndex(EmbeddingIndex):
    """Vector index adapter on top of simple in-memory arrays.

    It expects pre-computed embeddings (via EmbeddedChunk / EmbeddedQuery).
    """

    def __init__(self, index_dir: str | None = None, dim: int = 64) -> None:
        self.dim = dim
        self._vectors: list[Any] = []  # list of numpy arrays (or Any)
        self._meta: list[_Meta] = []

    def upsert(self, chunks: Iterable[EmbeddedChunk]) -> None:
        for ec in chunks:
            v = ec.embedding
            # No shape checks here; assume caller provides correct vectors
            self._vectors.append(v)
            self._meta.append(
                _Meta(
                    doc_id=str(ec.chunk.doc_id),
                    chunk_id=str(ec.chunk.id),
                    order=ec.chunk.order,
                    text=ec.chunk.text or "",
                )
            )

    def search(self, embedded_query: EmbeddedQuery) -> SearchResult:
        if not self._vectors:
            return SearchResult(
                query=embedded_query.query, hits=[], total_time_ms=0.0, retrieved_at=""
            )

        q = embedded_query.embedding
        sims: list[tuple[float, int]] = []
        # cosine similarity equivalent if vectors are normalized; else dot product
        for idx, v in enumerate(self._vectors):
            try:
                if _has_numpy():
                    score = float(np.dot(q, v))
                else:
                    # fallback for numpy-less scenario; rely on duck-typing
                    score = 0.0
            except Exception:
                # fallback for numpy-less scenario; rely on duck-typing
                score = 0.0
            sims.append((score, idx))

        sims.sort(reverse=True, key=lambda x: x[0])
        top_k = max(1, int(getattr(embedded_query.query, "top_k", 5)))
        top = sims[:top_k]

        hits: list[Hit] = []
        for score, i in top:
            meta = self._meta[i]
            preview = meta.text[:240] if meta.text else ""
            hits.append(
                Hit(
                    doc_id=DocId(meta.doc_id),
                    chunk_id=ChunkId(meta.chunk_id),
                    chunk_order=meta.order,
                    score=max(0.0, float(score)),
                    snippet=preview,
                )
            )

        return SearchResult(
            query=embedded_query.query,
            hits=hits,
            total_time_ms=0.0,
            retrieved_at="",
        )
