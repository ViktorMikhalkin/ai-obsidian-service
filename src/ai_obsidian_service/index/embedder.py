from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Pure contract: text -> vector. No IO, no storage, no side effects."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1D vector for the given text. Must be deterministic for tests."""
        raise NotImplementedError
