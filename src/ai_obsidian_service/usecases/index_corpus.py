from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path

from ai_obsidian_service.core import Chunker, DocumentParser
from ai_obsidian_service.adapters.services.search_service import SearchService  # NEW

class IndexCorpus:
    """Walk a root directory and let SearchService index each file."""

    def __init__(
            self,
            parsers: Iterable[DocumentParser],
            chunker: Chunker,
            service: SearchService,             # CHANGED: вместо EmbeddingIndex
    ) -> None:
        self.parsers = list(parsers)
        self.chunker = chunker
        self.service = service

    def run(self, root: str) -> int:
        count = 0
        for path in Path(root).rglob("*"):
            if not path.is_file():
                continue
            spath = str(path)
            for p in self.parsers:
                if p.can_parse(spath):
                    count += self.service.index_path(spath)   # DELEGATE
                    break
        return count
