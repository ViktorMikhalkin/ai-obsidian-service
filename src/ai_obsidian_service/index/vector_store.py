from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from ai_obsidian_service.domain.models import EmbeddedChunk, SearchResult


class VectorStore(ABC):
    """Pure vector storage & search. No embedding logic here."""

    @abstractmethod
    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Insert or update embedded chunks."""

    @abstractmethod
    def search(self, query_vec: np.ndarray, top_k: int) -> SearchResult:
        """Return top_k most similar chunks for the query vector."""
