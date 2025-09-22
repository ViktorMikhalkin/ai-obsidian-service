from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery, SearchResult, Hit, DocId, ChunkId
from ai_obsidian_service.ports.interfaces import EmbeddingIndex

def _ensure_np():
    if np is None:
        raise RuntimeError("NumPy is required for FaissIndex fallback embedding.")

def _embed_text(text: str, dim: int = 64):
    """Deterministic lightweight embedding without external deps."""
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
    If FAISS becomes available, we can swap internals transparently.
    """
    def __init__(self, index_dir: str | None = None, dim: int = 64) -> None:
        self.dim = dim
        # Fix: Add proper type annotation for _vectors
        self._vectors: list[np.ndarray] = []
        self._meta: List[_Meta] = []

    def upsert(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        """Add or update embedded chunks in the index."""
        _ensure_np()
        for chunk in embedded_chunks:
            # Use the pre-computed embedding from EmbeddedChunk
            self._vectors.append(chunk.embedding)
            chunk_id = f"{chunk.doc_id}#{chunk.order}"
            self._meta.append(_Meta(
                doc_id=str(chunk.doc_id),
                chunk_id=chunk_id,
                order=chunk.order,
                text=chunk.text
            ))

    def search(self, embedded_query: EmbeddedQuery) -> SearchResult:
        """Search for similar chunks using the embedded query."""
        if not self._vectors:
            return SearchResult(hits=[], total_time_ms=0.0, retrieved_at="")

        _ensure_np()

        # Use the pre-computed embedding from EmbeddedQuery
        query_vector = embedded_query.embedding

        # Calculate similarities
        sims = []
        for idx, vector in enumerate(self._vectors):
            # Compute cosine similarity
            score = float(np.dot(query_vector, vector))
            sims.append((score, idx))

        # Sort by similarity score (descending)
        sims.sort(reverse=True, key=lambda x: x[0])

        # Get top k results
        top_k = getattr(embedded_query, 'top_k', 5)  # Default to 5 if not specified
        top = sims[:max(1, int(top_k))]

        # Convert to Hit objects
        hits: List[Hit] = []
        for score, i in top:
            meta = self._meta[i]
            preview = meta.text[:240] if meta.text else ""
            hits.append(Hit(
                doc_id=DocId(meta.doc_id),
                chunk_id=ChunkId(meta.chunk_id),
                chunk_order=meta.order,
                score=max(0.0, score),  # Ensure non-negative score
                snippet=preview
            ))

        return SearchResult(
            hits=hits,
            total_time_ms=0.0,  # Could add actual timing measurement here
            retrieved_at=""     # Could add actual timestamp here
        )