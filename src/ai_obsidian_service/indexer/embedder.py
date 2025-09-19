import os

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        dtype: str = "fp32",
    ):
        # Override device if environment forces CPU mode
        # This allows CPU-only environments to work regardless of config.yml settings
        if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
            device = "cpu"
        if os.environ.get("TORCH_DEVICE") == "cpu":
            device = "cpu"

        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

        # Map dtype strings to numpy dtypes
        self._dtype_mapping = {
            "fp32": np.float32,
            "float32": np.float32,
            "fp16": np.float16,
            "float16": np.float16,
        }
        self.numpy_dtype = self._dtype_mapping.get(dtype, np.float32)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """Get vectors for the list of texts."""
        raw = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.array(raw, dtype=self.numpy_dtype)
