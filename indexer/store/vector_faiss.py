from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import faiss, numpy as np

@dataclass
class FaissIndex:
    dim: int
    path: Path

    def __post_init__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def add(self, embeddings: np.ndarray):
        emb = self._normalize(embeddings.astype(np.float32))
        self.index.add(emb)

    def search(self, query: np.ndarray, top_k: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1:
            query = query[None, :]
        q = self._normalize(query.astype(np.float32))
        return self.index.search(q, top_k)

    def save(self):
        faiss.write_index(self.index, str(self.path))
