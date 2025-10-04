from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ai_obsidian_service.domain.models import (
    Chunk,
    Document,
    EmbeddedChunk,
    EmbeddedQuery,
    Query,
    SearchResult,
)


class EnhancedEmbeddingIndex:
    """
    Enhanced embedding index with incremental updates and deduplication.

    Features:
    - Document-level change detection (via doc_hash)
    - Chunk deduplication (optional)
    - Batch processing for efficiency
    - Statistics tracking
    - Compatible with existing EmbeddingIndex interface

    Usage:
        index = EnhancedEmbeddingIndex(
            embedder=embedder,
            store=store,
            chunker=chunker,
            enable_dedup=True,
        )

        # Check if document needs reindexing
        if index.needs_reindex(doc):
            result = index.index_document(doc, force=False)

        # Save registry for incremental updates
        index.save_registry("./data/index/doc_registry.json")
    """

    def __init__(self, *, embedder, store, chunker, enable_dedup: bool = False) -> None:
        self.embedder = embedder
        self.store = store
        self.chunker = chunker
        self.enable_dedup = enable_dedup

        # Track indexed documents: {doc_path -> doc_hash}
        self._doc_registry: dict[str, str] = {}

        # Deduplication tracking (if enabled)
        self._chunk_hashes: set[str] | None = set() if enable_dedup else None

    def _compute_text_hash(self, text: str) -> str:
        """Compute normalized hash of text for deduplication."""
        normalized = " ".join(text.lower().strip().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def needs_reindex(self, doc: Document) -> bool:
        """
        Check if document needs reindexing based on content hash.

        Args:
            doc: Document to check

        Returns:
            True if document changed or is new
        """
        # Use path as registry key for duplicate detection
        doc_path = str(getattr(doc, "path", getattr(doc, "id", "")))
        meta: dict[str, Any] = getattr(doc, "metadata", None) or {}
        raw_hash = meta.get("doc_hash")
        if isinstance(raw_hash, str) and raw_hash:
            doc_hash: str = raw_hash
        else:
            # No hash (or not a string) in metadata — compute it
            doc_hash = self._compute_text_hash(getattr(doc, "text", "") or "")

        if not doc_hash:
            # No hash in metadata, compute it
            doc_hash = self._compute_text_hash(doc.text or "")

        # Check if changed
        stored_hash = self._doc_registry.get(doc_path)
        return stored_hash != doc_hash

    def _filter_duplicates(self, chunks: list[Chunk]) -> list[Chunk]:
        """Filter out duplicate chunks based on text hash."""
        if not self.enable_dedup or self._chunk_hashes is None:
            return chunks

        unique_chunks = []
        for chunk in chunks:
            text_hash = self._compute_text_hash(chunk.text or "")

            if text_hash not in self._chunk_hashes:
                self._chunk_hashes.add(text_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        """
        Embed and store chunks (with optional deduplication).

        Args:
            chunks: Sequence of chunks to index
        """
        if not chunks:
            return

        # Filter duplicates if enabled
        unique_chunks = self._filter_duplicates(list(chunks))

        if not unique_chunks:
            return

        # Batch embed
        texts = [c.text or "" for c in unique_chunks]

        # Check if embedder supports E5 prefix
        model_name = getattr(self.embedder, "model_name", "")
        use_e5_prefix = "e5" in model_name.lower()

        # Embed with prefix support (if available)
        if use_e5_prefix:
            try:
                vecs = self.embedder.embed(texts, prefix="passage: ")
            except TypeError:
                # Embedder doesn't support prefix parameter
                vecs = self.embedder.embed(texts)
        else:
            vecs = self.embedder.embed(texts)

        # Ensure 2D array
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        # Create embedded chunks
        emb_chunks = [
            EmbeddedChunk(chunk=c, embedding=v)
            for c, v in zip(unique_chunks, vecs, strict=False)
        ]

        # Store
        self.store.upsert(emb_chunks)

    def index_document(self, doc: Document, *, force: bool = False) -> dict:
        """
        Index a single document with incremental update support.

        Args:
            doc: Document to index
            force: Force reindexing even if unchanged

        Returns:
            Dictionary with statistics: {indexed, skipped, chunks, changed, doc_id, doc_path}
        """
        # Use path as registry key for duplicate detection
        doc_path = str(getattr(doc, "path", getattr(doc, "id", "")))
        doc_id = str(getattr(doc, "id", doc_path))

        # Safe extraction of doc_hash
        meta: dict[str, Any] = getattr(doc, "metadata", None) or {}
        raw_hash = meta.get("doc_hash")
        if isinstance(raw_hash, str) and raw_hash:
            doc_hash: str = raw_hash
        else:
            doc_text = getattr(doc, "text", "") or ""
            doc_hash = self._compute_text_hash(doc_text)

        # Check if needs reindexing
        if not force and not self.needs_reindex(doc):
            return {
                "indexed": False,
                "skipped": True,
                "chunks": 0,
                "changed": False,
                "doc_id": doc_id,
                "doc_path": doc_path,
            }

        # Split into chunks
        chunks = self.chunker.split(doc)

        if not chunks:
            return {
                "indexed": False,
                "skipped": False,
                "chunks": 0,
                "changed": True,
                "doc_id": doc_id,
                "doc_path": doc_path,
                "error": "No chunks generated",
            }

        # Embed and store
        self.upsert(chunks)

        # Update registry with path as key
        self._doc_registry[doc_path] = doc_hash

        return {
            "indexed": True,
            "skipped": False,
            "chunks": len(chunks),
            "changed": True,
            "doc_id": doc_id,
            "doc_path": doc_path,
        }

    def index_documents_batch(
        self,
        docs: Sequence[Document],
        *,
        force: bool = False,
        show_progress: bool = True,
    ) -> dict:
        """
        Index multiple documents efficiently.

        Args:
            docs: Sequence of documents to index
            force: Force reindexing all documents
            show_progress: Show progress information

        Returns:
            Dictionary with batch statistics
        """
        if not docs:
            return {"total": 0, "indexed": 0, "skipped": 0, "chunks": 0}

        total = len(docs)
        indexed_count = 0
        skipped_count = 0
        total_chunks = 0
        errors = []

        for i, doc in enumerate(docs, 1):
            if show_progress and i % 10 == 0:
                print(f"Processing {i}/{total} documents...")

            try:
                result = self.index_document(doc, force=force)

                if result["indexed"]:
                    indexed_count += 1
                    total_chunks += result["chunks"]
                elif result["skipped"]:
                    skipped_count += 1

                if "error" in result:
                    errors.append(
                        {
                            "doc_id": result["doc_id"],
                            "doc_path": result.get("doc_path", "unknown"),
                            "error": result["error"],
                        }
                    )

            except Exception as e:
                doc_path = doc.path if hasattr(doc, "path") else str(doc.id)
                errors.append(
                    {
                        "doc_id": str(doc.id),
                        "doc_path": doc_path,
                        "error": str(e),
                    }
                )

        return {
            "total": total,
            "indexed": indexed_count,
            "skipped": skipped_count,
            "chunks": total_chunks,
            "errors": errors if errors else None,
        }

    def search_text(self, text: str, *, top_k: int = 5) -> SearchResult:
        """
        Search for similar chunks using text query.

        Args:
            text: Query text
            top_k: Number of results to return

        Returns:
            SearchResult with hits
        """
        t0 = time.perf_counter()

        # Check if embedder supports E5 prefix
        model_name = getattr(self.embedder, "model_name", "")
        use_e5_prefix = "e5" in model_name.lower()

        # Embed query with prefix support
        if use_e5_prefix:
            try:
                vec = self.embedder.embed([text], prefix="query: ")
            except TypeError:
                vec = self.embedder.embed([text])
        else:
            vec = self.embedder.embed([text])

        eq = EmbeddedQuery(text=text, vector=np.asarray(vec[0], dtype=np.float32))

        # Search store
        store_result = self.store.search(eq.vector, top_k=int(top_k))

        return SearchResult(
            query=Query(text=text, top_k=int(top_k)),
            hits=store_result.hits,
            total_time_ms=round((time.perf_counter() - t0) * 1000.0, 3),
            retrieved_at=datetime.utcnow(),
        )

    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            "documents": len(self._doc_registry),
            "chunks": self.store.count if hasattr(self.store, "count") else 0,
            "deduplication_enabled": self.enable_dedup,
            "unique_chunk_hashes": len(self._chunk_hashes) if self._chunk_hashes else 0,
        }

    def save_registry(self, path: str | Path) -> None:
        """Save document registry for incremental updates."""
        registry_path = Path(path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(self._doc_registry, f, ensure_ascii=False, indent=2)

    def load_registry(self, path: str | Path) -> None:
        """Load document registry from previous run."""
        registry_path = Path(path)
        if not registry_path.exists():
            return

        with open(registry_path, encoding="utf-8") as f:
            self._doc_registry = json.load(f)
