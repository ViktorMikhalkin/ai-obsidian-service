from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from ai_obsidian_service.core import Chunk, ChunkId, DocId, EmbeddingIndex, Hit, Query


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
        self._vectors = []  # list[np.ndarray]
        self._meta: list[_Meta] = []

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        _ensure_np()
        for c in chunks:
            v = _embed_text(c.text, self.dim)
            self._vectors.append(v)
            chunk_id = f"{c.doc_id}#{c.order}"
            self._meta.append(
                _Meta(
                    doc_id=str(c.doc_id), chunk_id=chunk_id, order=c.order, text=c.text
                )
            )

    def search(self, query: Query) -> list[Hit]:
        if not self._vectors:
            return []
        _ensure_np()
        q = _embed_text(query.text, self.dim)
        sims = []
        for idx, v in enumerate(self._vectors):
            score = float((q * v).sum())
            sims.append((score, idx))
        sims.sort(reverse=True, key=lambda x: x[0])
        top = sims[: max(1, int(query.top_k))]
        hits: list[Hit] = []
        for score, i in top:
            m = self._meta[i]
            preview = m.text[:240] if m.text else ""
            hits.append(
                Hit(
                    doc_id=DocId(m.doc_id),
                    chunk_id=ChunkId(m.chunk_id),
                    chunk_order=m.order,
                    score=score if score > 0 else 0.0,
                    snippet=preview,
                )
            )
        return hits
