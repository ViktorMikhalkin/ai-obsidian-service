from __future__ import annotations
from collections.abc import Iterable
from pathlib import Path

from ai_obsidian_service.core import Chunker, DocumentParser
from ai_obsidian_service.adapters.services.search_service import SearchService


class IndexCorpus:
    """
    Bulk indexing of a directory: parser selection, parsing and delegation of indexing to SearchService.
    """

    def __init__(
            self,
            parsers: Iterable[DocumentParser],
            chunker: Chunker,              # keep in signature for DI consistency, though not used internally
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
        return count