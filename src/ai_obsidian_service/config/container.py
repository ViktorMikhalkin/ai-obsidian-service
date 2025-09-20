from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
else:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[misc]

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.index.faiss_index import FaissIndex
from ai_obsidian_service.adapters.parsers import default_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Chunk, Query
from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery


def _ensure_np():
    if np is None:
        raise RuntimeError("NumPy is required for embedder.")


def _embed_text(text: str, dim: int = 64) -> np.ndarray:
    """Simple fallback embedding function."""
    _ensure_np()
    vec = np.zeros(dim, dtype=float)
    if not text:
        return vec.astype(np.float32)
    for i, ch in enumerate(text):
        vec[(ord(ch) + i) % dim] += 1.0
    n = np.linalg.norm(vec)
    if n > 0:
        vec = vec / n
    return vec.astype(np.float32)


class SimpleEmbedder:
    """Simple embedder using character-based hashing."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        return _embed_text(text, self.dim)

    def embed_chunk(self, chunk: Chunk) -> EmbeddedChunk:
        embedding = self.embed_text(chunk.text)
        return EmbeddedChunk(chunk=chunk, embedding=embedding)

    def embed_query(self, query: Query) -> EmbeddedQuery:
        embedding = self.embed_text(query.text)
        return EmbeddedQuery(query=query, embedding=embedding)


def build_search_service(index_dir: str | None = None) -> SearchService:
    return SearchService(
        parsers=default_parsers(),
        chunker=SimpleChunker(max_chars=1000, overlap=100),
        index=FaissIndex(index_dir=index_dir, dim=64),
        embedder=SimpleEmbedder(dim=64),
    )
