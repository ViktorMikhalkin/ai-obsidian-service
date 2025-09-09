from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=False, show_progress_bar=False)
        import numpy as np
        return np.array(vecs, dtype="float32")
