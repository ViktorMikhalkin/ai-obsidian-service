from __future__ import annotations

from dataclasses import dataclass

from ai_obsidian_service.core import Document, DocumentParser
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.domain.models import SearchResult


@dataclass(slots=True)
class SearchService:
    """Application service that delegates to EmbeddingIndex and parser.

    It does not perform embedding itself and does not know about VectorStore details.
    """
    index: EmbeddingIndex
    parser: DocumentParser

    def index_path(self, path: str) -> int:
        doc = self.parser.parse(path)
        return self.index.index_document(doc)

    def search_text(self, text: str, top_k: int = 5) -> SearchResult:
        return self.index.search(text=text, top_k=top_k)
