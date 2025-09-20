from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[misc]

from ai_obsidian_service.core import ChunkId, DocId, EmbeddingIndex, Hit
from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery, SearchResult


def _ensure_np():
    if np is None:
        raise RuntimeError("NumPy is required for FaissIndex fallback embedding.")


def _embed_text(text: str, dim: int = 64):
    """Deterministic lightweight embedding without external deps.
    Maps characters to a fixed-size vector by hashing codepoints.
    """
    _ensure_np()
    vec = np.zeros(dim, dtype=float)
    if not text:
        return vec
    for i, ch in enumerate(text):
        vec[(ord(ch) + i) % dim] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec


@dataclass(frozen=True)
class _Meta:
    doc_id: str
    chunk_id: str
    order: int
    text: str


class FaissIndex(EmbeddingIndex):
    """Adapter implementing EmbeddingIndex over a simple NumPy fallback.
    If FAISS is available, it can be wired here later transparently.
    """

    def __init__(self, index_dir: str | None = None, dim: int = 64) -> None:
        self.dim = dim
        self._vectors: list[np.ndarray] = []
        self._meta: list[_Meta] = []

    def upsert(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        chunks_list = list(embedded_chunks)
        if not chunks_list:
            return
        _ensure_np()

        for ec in chunks_list:
            chunk = ec.chunk
            # Use the provided embedding or fall back to our simple embedding
            if hasattr(ec, "embedding") and ec.embedding is not None:
                v = ec.embedding
            else:
                v = _embed_text(chunk.text, self.dim)

            self._vectors.append(v)
            chunk_id = f"{chunk.doc_id}#{chunk.order}"
            self._meta.append(
                _Meta(
                    doc_id=str(chunk.doc_id),
                    chunk_id=chunk_id,
                    order=chunk.order,
                    text=chunk.text,
                )
            )

    def search(self, embedded_query: EmbeddedQuery) -> SearchResult:
        start_time = time.perf_counter()

        if not self._vectors:
            return SearchResult(
                query=embedded_query.query,
                hits=[],
                total_time_ms=0.0,
                retrieved_at=datetime.now().isoformat(),
            )

        _ensure_np()

        # Use the provided query embedding or fall back to our simple embedding
        if (
            hasattr(embedded_query, "embedding")
            and embedded_query.embedding is not None
        ):
            q = embedded_query.embedding
        else:
            q = _embed_text(embedded_query.query.text, self.dim)

        sims = []
        for idx, v in enumerate(self._vectors):
            score = float((q * v).sum())
            sims.append((score, idx))
        sims.sort(reverse=True, key=lambda x: x[0])
        top = sims[: max(1, int(embedded_query.query.top_k))]

        hits: list[Hit] = []
        for score, i in top:
            m = self._meta[i]
            preview = m.text[:240] if m.text else ""
            hits.append(
                Hit(
                    chunk_id=ChunkId(m.chunk_id),
                    doc_id=DocId(m.doc_id),
                    chunk_order=m.order,
                    score=score if score > 0 else 0.0,
                    snippet=preview,
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResult(
            query=embedded_query.query,
            hits=hits,
            total_time_ms=elapsed_ms,
            retrieved_at=datetime.now().isoformat(),
        )
