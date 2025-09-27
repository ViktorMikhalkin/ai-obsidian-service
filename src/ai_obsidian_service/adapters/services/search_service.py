from __future__ import annotations

from typing import Optional, Sequence, Iterable, List

from ai_obsidian_service.domain.models import Hit, SearchResult


class SearchService:
    """
    Thin service over EmbeddingIndex + parsers + (opt.) BM25 re-ranker.

    - parsers: list of parser adapters; first .can_parse(path) wins
    - index  : EmbeddingIndex with .index_document(Document) and .search_text(str, top_k)
    - reranker (optional): must expose rerank(query: str, hits: list[Hit], limit: int) -> Iterable[Hit]
    """

    def __init__(
            self,
            *,
            index,
            parsers: Sequence[object],
            reranker: object | None = None,
            rerank_topn: int = 50,
    ) -> None:
        self.index = index
        self.parsers = list(parsers)
        self.reranker = reranker
        self.rerank_topn = int(rerank_topn)

    # ---------- indexing ----------

    def _select_parser(self, path: str):
        for p in self.parsers:
            try:
                if p.can_parse(path):
                    return p
            except Exception:
                # keep robust; broken parser shouldn't crash the whole run
                continue
        return None

    def index_path(self, path: str) -> int:
        parser = self._select_parser(path)
        if parser is None:
            return 0
        doc = parser.parse(path)
        if doc is None:
            return 0
        return self.index.index_document(doc)

    # ---------- search ----------

    def _filter_by_collection(self, hits: Sequence[Hit], collection: Optional[str]) -> list[Hit]:
        if not collection:
            return list(hits)
        want = collection
        out: list[Hit] = []
        for h in hits:
            meta = (h.metadata or (h.chunk.metadata if (h.chunk and getattr(h.chunk, "metadata", None)) else None)) or {}
            if meta.get("collection") == want:
                out.append(h)
        return out

    def search_text(self, text: str, top_k: int = 5, collection: str | None = None) -> SearchResult:
        # 1) retrieve topN from vector index
        candidates_k = max(self.rerank_topn, top_k) if self.reranker is not None else top_k
        result = self.index.search_text(text, top_k=int(candidates_k))
        hits = result.hits

        # 2) (optional) BM25 rerank
        if self.reranker is not None and hits:
            reranked: Iterable[Hit] = self.reranker.rerank(query=text, hits=hits[:candidates_k], limit=top_k)
            hits = list(reranked)

        # 3) post-filter by collection + cut to top_k
        hits = self._filter_by_collection(hits, collection)[: int(top_k)]
        result.hits = hits
        return result

    # ---------- API helpers ----------

    def resolve_meta(self, *, chunk_id: str) -> dict:
        # best-effort; memory store doesn’t keep a reverse map
        return {}
