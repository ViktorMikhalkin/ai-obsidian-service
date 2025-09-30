from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_obsidian_service.index.embedder import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Production-grade embedder based on `sentence-transformers`.

    Notes:
    - Returns L2-normalized float32 vectors (good for IP/cosine).
    - Deterministic for a given model and text.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> np.ndarray:
        # encode outputs a 1D array when `convert_to_numpy=True` and `normalize_embeddings=True`
        vec = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # suitable for IndexFlatIP (cosine similarity)
        )
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec)
        result: np.ndarray = vec.astype(np.float32, copy=False)
        return result
