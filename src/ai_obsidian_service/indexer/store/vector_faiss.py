from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable, Sequence
from datetime import datetime as _dt
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ai_obsidian_service.domain.models import (
    ChunkId,
    DocId,
    EmbeddedChunk,
    EmbeddedQuery,
    Hit,
    SearchResult,
)


def _ts(msg: str):
    print(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")


class FaissIndex:
    """FAISS-based vector index implementing VectorIndex and EmbeddingIndex protocols."""

    def __init__(self, dim: int, path: Path):
        self.dim = dim
        self.path = Path(path)
        self.index = faiss.IndexFlatIP(dim)
        self._chunk_metadata: list[dict[str, Any]] = []

    # Legacy method - kept for backward compatibility
    def add(self, vecs: np.ndarray):
        """Add vectors to the index (legacy method)."""
        self.index.add(vecs)

    # VectorIndex protocol methods
    def add_vectors(self, vectors: np.ndarray, metadata: Sequence[dict]) -> None:
        """Add vectors with associated metadata."""
        if vectors.shape[0] != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")

        self.index.add(vectors.astype(np.float32))
        self._chunk_metadata.extend(metadata)

    def search_vectors(
        self, query_vector: np.ndarray, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for similar vectors, returning indices and scores."""
        query_vec = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vec, top_k)
        return indices[0], scores[0]

    # EmbeddingIndex protocol methods
    def upsert(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        """Add embedded chunks to the index."""
        chunks_list = list(embedded_chunks)
        if not chunks_list:
            return

        # Extract vectors and metadata
        vectors = np.array([ec.embedding for ec in chunks_list], dtype=np.float32)
        metadata = []

        for ec in chunks_list:
            chunk = ec.chunk
            metadata.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "order": chunk.order,
                    "text": chunk.text,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char or len(chunk.text),
                    "metadata": chunk.metadata or {},
                }
            )

        self.add_vectors(vectors, metadata)

    def search(self, embedded_query: EmbeddedQuery) -> SearchResult:
        """Search for similar chunks using embedded query."""
        start_time = time.perf_counter()

        indices, scores = self.search_vectors(
            embedded_query.embedding, embedded_query.query.top_k
        )

        # Convert results to Hit objects
        hits = []
        for idx, score in zip(indices, scores, strict=False):
            if idx < len(self._chunk_metadata) and idx >= 0:
                meta = self._chunk_metadata[idx]
                hit = Hit(
                    chunk_id=ChunkId(meta["chunk_id"]),
                    doc_id=DocId(meta["doc_id"]),
                    chunk_order=meta["order"],
                    score=float(score),
                    snippet=self._create_snippet(meta["text"]),
                    start_char=meta["start_char"],
                    end_char=meta["end_char"],
                    metadata=meta["metadata"],
                )
                hits.append(hit)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResult(
            query=embedded_query.query,
            hits=hits,
            total_time_ms=elapsed_ms,
            retrieved_at=_dt.now().isoformat(),
        )

    def _create_snippet(self, text: str, max_length: int = 200) -> str:
        """Create a snippet from the full text."""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    # Persistence methods
    def save(self):
        """Save index to disk."""
        _ts(f"[faiss] saving index → {self.path}")
        t0 = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix="faiss.", suffix=".tmp"
        )
        os.close(fd)
        try:
            faiss.write_index(self.index, tmp_path)
            os.replace(tmp_path, str(self.path))

            # Also save metadata
            metadata_path = self.path.with_suffix(".metadata.json")
            self.save_metadata(metadata_path)

        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        dur = time.perf_counter() - t0
        try:
            size = os.path.getsize(self.path)
        except Exception:
            size = -1
        _ts(f"[faiss] saved in {dur:.1f}s, size={size / 1e6:.2f} MB")

    @classmethod
    def load(cls, path: Path):
        """Load index from disk."""
        index = faiss.read_index(str(path))
        fi = cls(index.d, path)
        fi.index = index

        # Also load metadata if it exists
        metadata_path = path.with_suffix(".metadata.json")
        if metadata_path.exists():
            fi.load_metadata(metadata_path)

        return fi

    def save_metadata(self, metadata_path: Path):
        """Save chunk metadata separately."""
        with open(metadata_path, "w") as f:
            json.dump(self._chunk_metadata, f, indent=2)

    def load_metadata(self, metadata_path: Path):
        """Load chunk metadata."""
        if metadata_path.exists():
            with open(metadata_path) as f:
                self._chunk_metadata = json.load(f)

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return int(self.index.ntotal)
