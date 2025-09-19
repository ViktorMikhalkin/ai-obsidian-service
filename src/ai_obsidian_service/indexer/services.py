import json
from pathlib import Path

import yaml

from ai_obsidian_service.indexer.embedder import Embedder
from ai_obsidian_service.indexer.store.vector_faiss import FaissIndex


class IndexerService:
    def __init__(self, config_path="config.yaml"):
        self.cfg = self._load_config(config_path)
        self.index_dir = Path(self.cfg.get("index_dir", "index"))
        self.fa = self._load_index()
        self.metas = self._load_metas()
        self.dim = self._load_dim()
        self.embed = self._init_embedder()

    def _load_config(self, config_path):
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_index(self):
        idx_path = self.index_dir / "faiss.index"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing FAISS index at {idx_path}")
        return FaissIndex.load(idx_path)

    def _load_metas(self):
        meta_path = self.index_dir / "index.jsonl"
        metas = []
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    metas.append(json.loads(line))
                except Exception:
                    continue
        return metas

    def _load_dim(self):
        dim_path = self.index_dir / "dim.txt"
        return int(Path(dim_path).read_text().strip())

    def _init_embedder(self):
        emb = self.cfg.get("embeddings", {}) or {}
        model = emb.get("model", "intfloat/multilingual-e5-small")
        device = emb.get("device", "cpu")
        dtype = emb.get("dtype", "fp32")
        return Embedder(model_name=model, device=device, dtype=dtype)
