from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        device: str = "cpu",
        dtype: str = "fp32",
    ):
        # dtype пока просто хранится, если решите управлять torch.dtype вручную
        self.device = device
        self.dtype = dtype
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        """Получить векторы для списка текстов."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
