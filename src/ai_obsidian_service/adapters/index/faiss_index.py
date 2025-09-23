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


def _py_dot(a: list[float] | Any, b: list[float] | Any) -> float:
    """Pure-Python dot product as a fallback when NumPy is unavailable."""
    try:
        return float(sum((float(x) * float(y) for x, y in zip(a, b, strict=False))))
    except Exception:
        return 0.0


def _has_numpy() -> bool:
    try:
        import numpy as _np  # noqa: F401

        return True
    except Exception:
        return False


if _has_numpy():
    import numpy as np  # Remove the unused type: ignore comment
else:
    np = None  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
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
            if _has_numpy():
                self._vectors.append(v)
            else:
                # store as plain list of floats
                self._vectors.append([float(x) for x in v])
            self._meta.append(
                _Meta(
                    doc_id=str(ec.chunk.doc_id),
                    chunk_id=str(ec.chunk.id),
                    order=int(ec.chunk.order),
                    text=str(ec.chunk.text or ""),
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
                    score = _py_dot(q, v)
            except Exception:
                score = 0.0
            sims.append((score, idx))

        sims.sort(reverse=True, key=lambda x: x[0])
        top_k = max(1, int(getattr(embedded_query.query, "top_k", 5)))
        top = sims[:top_k]

        hits: list[Hit] = []
        for score, idx in top:
            meta = self._meta[idx]
            preview = meta.text[:240]
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