from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ai_obsidian_service.core import Document, DocumentParser
from ai_obsidian_service.domain.models import Hit, SearchResult
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.rerank.bm25 import BM25Reranker


@dataclass(slots=True)
class SearchService:
    """
    SearchService that supports multiple parsers.
    - If constructed with `parsers`, it will select a parser by can_parse(path).
    - For backward-compatibility, `parser=` is still accepted and wrapped as a single-item list.
    """
    index: EmbeddingIndex
    parsers: Sequence[DocumentParser]
    reranker: BM25Reranker | None = None
    rerank_topn: int = 50

    # Backward-compatible signature: allow `parser=` OR `parsers=`.
    def __init__(
            self,
            index: EmbeddingIndex,
            parser: DocumentParser | None = None,
            parsers: Sequence[DocumentParser] | None = None,
            reranker: BM25Reranker | None = None,
            rerank_topn: int = 50,
            **_: Any,
    ) -> None:
        self.index = index
        if parsers is not None and len(parsers) > 0:
            self.parsers = tuple(parsers)
        elif parser is not None:
            self.parsers = (parser,)
        else:
            raise ValueError("SearchService requires at least one DocumentParser (parser= or parsers=).")
        self.reranker = reranker
        self.rerank_topn = int(rerank_topn)

    # ---------- indexing ----------

    def index_document(self, document: Document) -> int:
        return self.index.index_document(document)

    def _select_parser(self, path: str) -> DocumentParser:
        for p in self.parsers:
            try:
                if p.can_parse(path):
                    return p
            except Exception:
                # Be robust to parser-specific issues when probing
                continue
        raise ValueError(f"No parser available for path: {path}")

    def index_path(self, path: str) -> int:
        parser = self._select_parser(path)
        doc = parser.parse(path)
        return self.index.index_document(doc)

    # ---------- search ----------

    def _post_filter_collection(self, hits: Sequence[Hit], collection: str | None) -> list[Hit]:
        if not collection:
            return list(hits)
        out: list[Hit] = []
        for h in hits:
            # prefer h.metadata; fallback to chunk.metadata if present
            meta = (h.metadata or (h.chunk.metadata if h.chunk else None)) or {}
            if meta.get("collection") == collection:
                out.append(h)
        return out

    def search_text(self, text: str, top_k: int = 5, collection: str | None = None) -> SearchResult:
        candidates_k = max(top_k, self.rerank_topn if self.reranker else top_k)
        result = self.index.search(text, top_k=candidates_k)
        hits = self._post_filter_collection(result.hits, collection)

        if self.reranker and hits:
            pool = hits[: self.rerank_topn]
            reranked = self.reranker.rerank(text, pool)
            rest = [h for h in hits if h not in pool]
            hits = list(reranked) + rest

        result.hits = hits[:top_k]
        return result

    # ---------- misc ----------

    def resolve_meta(self, chunk_id: str) -> dict[str, Any]:  # pragma: no cover
        return {}

    def shutdown(self) -> None:  # pragma: no cover
        return None
