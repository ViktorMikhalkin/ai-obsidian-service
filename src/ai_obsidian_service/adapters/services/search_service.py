
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai_obsidian_service.core import Document, DocumentParser
from ai_obsidian_service.domain.models import Hit, SearchResult
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.rerank.bm25 import BM25Reranker


@dataclass(slots=True)
class SearchService:
    """Application service that delegates to EmbeddingIndex and parser."""
    index: EmbeddingIndex
    parser: DocumentParser
    reranker: BM25Reranker | None = None
    rerank_topn: int = 50

    def index_document(self, document: Document) -> int:
        return self.index.index_document(document)

    def index_path(self, path: str) -> int:
        doc = self.parser.parse(path)
        return self.index.index_document(doc)

    def _post_filter_collection(self, hits: Sequence[Hit], collection: str | None) -> list[Hit]:
        if not collection:
            return list(hits)
        out: list[Hit] = []
        for h in hits:
            meta = getattr(h.chunk, "metadata", None) or {}
            if meta.get("collection") == collection:
                out.append(h)
        return out

    def search_text(self, text: str, top_k: int = 5, collection: str | None = None) -> SearchResult:
        candidates_k = max(top_k, self.rerank_topn if self.reranker else top_k)
        result = self.index.search(text=text, top_k=candidates_k)
        hits = self._post_filter_collection(result.hits, collection)

        if self.reranker is not None and hits:
            pool = hits[: self.rerank_topn]
            reranked = self.reranker.rerank(text, pool)
            rest = [h for h in hits if h not in pool]
            hits = list(reranked) + rest

        result.hits = hits[:top_k]
        return result
