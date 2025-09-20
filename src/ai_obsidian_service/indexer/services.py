import json
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import yaml

from ai_obsidian_service.domain.models import (
    ChunkId,
    DocId,
    EmbeddedQuery,
    Hit,
    Query,
    SearchResult,
)
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
        self.llm_client: object | None = None  # For future LLM integration

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_index(self):
        """Load FAISS index from disk."""
        idx_path = self.index_dir / "faiss.index"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing FAISS index at {idx_path}")

        fa_index = FaissIndex.load(idx_path)

        # Load metadata if it exists in the new format
        metadata_path = idx_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            fa_index.load_metadata(metadata_path)

        return fa_index

    def _load_metas(self):
        """Load legacy metadata from JSONL file."""
        meta_path = self.index_dir / "index.jsonl"
        metas = []
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        metas.append(json.loads(line))
                    except Exception:
                        continue
        return metas

    def _load_dim(self):
        """Load embedding dimension from file."""
        dim_path = self.index_dir / "dim.txt"
        return int(Path(dim_path).read_text().strip())

    def _init_embedder(self):
        """Initialize embedder with configuration."""
        emb = self.cfg.get("embeddings", {}) or {}
        model = emb.get("model", "intfloat/multilingual-e5-small")
        device = emb.get("device", "cpu")
        dtype = emb.get("dtype", "fp32")
        return Embedder(model_name=model, device=device, dtype=dtype)

    def search_text(self, query_text: str, top_k: int = 5) -> SearchResult:
        """
        Search for text using the enhanced domain architecture.

        This method bridges between the old metadata format and new domain models.
        """
        # Create domain query
        query = Query(text=query_text, top_k=top_k)

        # Generate embedding
        embedding = self.embed.encode([query_text])[0]
        embedded_query = EmbeddedQuery(query=query, embedding=embedding)

        # If the FAISS index has the new search method, use it
        if hasattr(self.fa, "search") and hasattr(self.fa, "_chunk_metadata"):
            result = self.fa.search(embedded_query)
            return cast(SearchResult, result)

        # Otherwise, use legacy method with metadata conversion
        return self._legacy_search(embedded_query)

    def _legacy_search(self, embedded_query: EmbeddedQuery) -> SearchResult:
        """
        Legacy search method that works with old metadata format.
        """
        start_time = time.perf_counter()

        # Use the raw FAISS search
        query_vec = embedded_query.embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.fa.index.search(query_vec, embedded_query.query.top_k)

        # Convert legacy metadata to domain hits
        hits = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            # Convert numpy types to Python types explicitly
            idx_int: int = int(idx.item()) if hasattr(idx, "item") else int(idx)
            score_float: float = (
                float(score.item()) if hasattr(score, "item") else float(score)
            )

            if idx_int < len(self.metas) and idx_int >= 0:
                meta = self.metas[idx_int]

                # Extract information from legacy metadata
                doc_id = DocId(meta.get("path", f"doc_{idx_int}"))
                chunk_id = ChunkId(f"{doc_id}_chunk_{idx_int}")

                # Handle start_char and end_char - convert from potential JSON floats to ints
                start_char_value = meta.get("start_char", 0)
                start_char = 0 if start_char_value is None else int(start_char_value)

                end_char_value = meta.get("end_char")
                end_char = None if end_char_value is None else int(end_char_value)

                hit = Hit(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_order=idx_int,
                    score=score_float,
                    snippet=self._create_snippet(meta.get("text", "")),
                    start_char=start_char,
                    end_char=end_char,
                    metadata={
                        "kind": meta.get("kind", ""),
                        "preview": meta.get("preview", ""),
                    },
                )
                hits.append(hit)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResult(
            query=embedded_query.query,
            hits=hits,
            total_time_ms=elapsed_ms,
            retrieved_at=datetime.now().isoformat(),
        )

    def _create_snippet(self, text: str, max_length: int = 200) -> str:
        """Create a snippet from the full text."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def get_stats(self) -> dict:
        """Get index statistics."""
        try:
            total_chunks = len(self.metas)
            if hasattr(self.fa, "size"):
                total_chunks = max(total_chunks, self.fa.size)

            # Count unique documents
            unique_docs = set()
            for meta in self.metas:
                if "path" in meta:
                    unique_docs.add(meta["path"])

            # Get index file size
            idx_path = self.index_dir / "faiss.index"
            index_size_mb: float = 0.0
            if idx_path.exists():
                index_size_mb = idx_path.stat().st_size / (1024 * 1024)

            return {
                "total_chunks": total_chunks,
                "total_documents": len(unique_docs),
                "index_size_mb": round(index_size_mb, 2),
                "last_updated": datetime.now().isoformat(),
                "embedding_dim": self.dim,
                "model_name": getattr(self.embed, "model_name", "unknown"),
            }
        except Exception as e:
            return {"error": str(e)}

    def rebuild_index(self) -> dict:
        """
        Rebuild the search index.

        This is a placeholder - you'll need to implement the actual rebuild logic
        based on your document processing pipeline.
        """
        try:
            start_time = time.perf_counter()

            # Placeholder for rebuild logic
            # You would typically:
            # 1. Clear existing index
            # 2. Re-process all documents
            # 3. Re-generate embeddings
            # 4. Rebuild FAISS index

            # For now, just return current stats
            stats = self.get_stats()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return {
                "rebuild_time_ms": elapsed_ms,
                "stats": stats,
                "message": "Index rebuild completed (placeholder implementation)",
            }
        except Exception as e:
            return {"error": str(e)}

    # Legacy compatibility methods
    def search_legacy(self, query_text: str, top_k: int = 5):
        """
        Legacy search method for backward compatibility.
        Returns the old format expected by existing code.
        """
        embedding = self.embed.encode([query_text])[0]
        query_vec = embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self.fa.index.search(query_vec, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < len(self.metas):
                meta = self.metas[idx]
                results.append(
                    {
                        "id": int(idx),
                        "path": meta.get("path", ""),
                        "kind": meta.get("kind", ""),
                        "preview": meta.get("preview", meta.get("text", "")[:200]),
                        "score": float(score),
                    }
                )

        return results
