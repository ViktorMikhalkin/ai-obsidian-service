from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Chunker, DocumentParser


class IndexCorpus:
    """
    Bulk indexing of a directory: parser selection, parsing and delegation of indexing to SearchService.
    """

    def __init__(
        self,
        parsers: Iterable[DocumentParser],
        chunker: Chunker,  # keep in signature for DI consistency, though not used internally
        service: SearchService,
    ) -> None:
        self.parsers = list(parsers)
        self.service = service

    def run(self, root: str) -> int:
        count = 0
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            spath = str(path)
            for p in self.parsers:
                if p.can_parse(spath):
                    count += self.service.index_path(spath)
                    break

        # Persist to disk if using FAISS store
        self._persist_index()

        return count

    def _persist_index(self) -> None:
        """Save the index to disk if the store supports persistence."""
        try:
            store = self.service.index.store
            # Check if this is a FAISS store with save method
            if hasattr(store, "save"):
                index_dir = os.getenv("INDEX_DIR")
                if index_dir:
                    # Get model name from embedder if available
                    model_name = None
                    if hasattr(self.service.index, "embedder") and hasattr(
                        self.service.index.embedder, "model_name"
                    ):
                        model_name = self.service.index.embedder.model_name

                    store.save(index_dir, model_name=model_name)
        except Exception:
            # Best effort - don't fail the indexing if save fails
            pass
