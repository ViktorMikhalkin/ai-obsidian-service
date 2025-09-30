from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import datetime

import numpy as np

from ai_obsidian_service.domain.models import (
    Chunk,
    EmbeddedChunk,
    EmbeddedQuery,
    Query,
    SearchResult,
)


class EmbeddingIndex:
    def __init__(self, *, embedder, store, chunker) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker

    # raw chunks arrive from service -> embed them and store in the vector store
    def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text or "" for c in chunks]
        vecs = self.embedder.embed(texts)  # -> np.ndarray[float32] (N, D)
        emb_chunks = [EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vecs, strict=False)]
        self.store.upsert(emb_chunks)

    # index a single document (convenient shortcut)
    def index_document(self, doc) -> int:
        chunks = self.chunker.split(doc)
        self.upsert(chunks)
        return len(chunks)

    # text search: embed(query) -> store.search -> SearchResult
    def search_text(self, text: str, *, top_k: int = 5) -> SearchResult:
        t0 = time.perf_counter()
        vec = self.embedder.embed([text])
        eq = EmbeddedQuery(text=text, vector=np.asarray(vec[0], dtype=np.float32))
        store_result = self.store.search(eq.vector, top_k=int(top_k))
        return SearchResult(
            query=Query(text=text, top_k=int(top_k)),
            hits=store_result.hits,
            total_time_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            retrieved_at=datetime.utcnow(),
        )
