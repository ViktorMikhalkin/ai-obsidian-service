from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ai_obsidian_service.domain.models import (
    ChunkId,
    DocId,
    EmbeddedChunk,
    Hit,
    SearchResult,
)


def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@dataclass(slots=True)
class InMemoryVectorStore:
    """
    Simple in-memory vector store (no FAISS). Keeps embeddings and chunk metadata.
    """

    dim: int | None = None

    # IMPORTANT: with slots=True we must declare attributes as fields
    _vecs: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _chunks: list[EmbeddedChunk] = field(default_factory=list, init=False, repr=False)

    @property
    def count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._chunks)

    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if not chunks:
            return
        if self.dim is None:
            self.dim = int(len(chunks[0].embedding))
        block = np.asarray([c.embedding for c in chunks], dtype=np.float32)
        self._vecs.extend(list(block))
        self._chunks.extend(list(chunks))

    def search(self, query_vec: np.ndarray, top_k: int) -> SearchResult:
        if not self._vecs:
            return SearchResult(query=None, hits=[])

        q = np.asarray(query_vec, dtype=np.float32)
        sims: list[tuple[float, int]] = []
        for i, v in enumerate(self._vecs):
            sims.append((_dot(q, v), i))
        sims.sort(key=lambda t: t[0], reverse=True)
        sims = sims[: max(1, int(top_k))]

        hits: list[Hit] = []
        for score, idx in sims:
            emb_chunk = self._chunks[idx]
            ch = emb_chunk.chunk
            snippet = (ch.text or "")[:240]
            hits.append(
                Hit(
                    doc_id=DocId(str(ch.doc_id)),
                    chunk_id=ChunkId(str(ch.id)),
                    chunk_order=int(ch.order),
                    score=max(0.0, float(score)),
                    snippet=snippet,
                    chunk=ch,
                    metadata=(ch.metadata or {}),
                )
            )
        return SearchResult(query=None, hits=hits)
