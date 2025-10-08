from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_obsidian_service.index.embedder import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Production-grade embedder based on `sentence-transformers`.

    Notes:
    - Returns L2-normalized float32 vectors (good for IP/cosine).
    - Deterministic for a given model and text.
    - Supports both single-text and batch processing.
    - Optimized for GPU utilization with batching.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model identifier
            batch_size: Batch size for processing multiple texts
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self._model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.batch_size = batch_size

    def embed(self, text: str | list[str]) -> np.ndarray:
        """
        Embed single text or list of texts.

        Args:
            text: Single text string or list of text strings

        Returns:
            - For single text: 1D array of shape (embedding_dim,)
            - For list of texts: 2D array of shape (len(texts), embedding_dim)
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        if not texts:
            return np.array([], dtype=np.float32)

        # Batch encoding for better GPU utilization
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,  # suitable for IndexFlatIP (cosine similarity)
            show_progress_bar=False,
        )

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings)

        result: np.ndarray = embeddings.astype(np.float32, copy=False)

        # Return 1D array for single text (backward compatible)
        if is_single:
            return np.asarray(result[0], dtype=np.float32)

        return result

    @property
    def dim(self) -> int:
        """Get the embedding dimension."""
        return self._model.get_sentence_embedding_dimension()  # type: ignore[no-any-return]
