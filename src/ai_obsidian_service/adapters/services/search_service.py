from __future__ import annotations

from typing import List, Dict, Tuple, Any, cast

from ai_obsidian_service.core import (
    DocumentParser,
    Chunker,
    EmbeddingIndex,
    Document,
    Chunk,
    Query,
    Hit,
    DocId,
    ChunkId,
)
from ai_obsidian_service.domain.models import EmbeddedChunk, EmbeddedQuery

def _embed_text(text: str, dim: int = 64):
    """Deterministic lightweight embedding (works with or without NumPy)."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        vec = [0.0] * dim
        if text:
            for i, ch in enumerate(text):
                vec[(ord(ch) + i) % dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm if norm > 0 else 0.0 for v in vec]
    else:
        v = np.zeros(dim, dtype=float)
        if text:
            for i, ch in enumerate(text):
                v[(ord(ch) + i) % dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

class SearchService:
    """Orchestrates parsing → chunking → indexing and search over EmbeddingIndex."""

    def __init__(
        self,
        parsers: List[DocumentParser],
        chunker: Chunker,
        index: EmbeddingIndex,
    ) -> None:
        self.parsers = parsers
        self.chunker = chunker
        self.index = index
        # meta by (doc_id, chunk_order)
        self._meta: Dict[Tuple[str, int], Dict[str, Any]] = {}

    # ---------- Indexing ----------

    def index_document(self, doc: Document) -> int:
        """Index a parsed document and return number of chunks stored."""
        ck: Any = cast(Any, self.chunker)
        chunks: List[Chunk]
        if hasattr(ck, "chunk") and callable(getattr(ck, "chunk")):
            chunks = ck.chunk(doc)
        elif hasattr(ck, "chunk_document") and callable(getattr(ck, "chunk_document")):
            chunks = ck.chunk_document(doc)
        else:
            # Fallback: single full-text chunk
            doc_id_val = str(getattr(doc, "id", "doc"))
            text_val = str(getattr(doc, "text", ""))
            chunks = [
                Chunk(
                    id=ChunkId(f"{doc_id_val}#0"),
                    doc_id=DocId(doc_id_val),
                    order=0,
                    text=text_val,
                )
            ]

        dim = getattr(self.index, 'dim', 64)
        embedded: List[EmbeddedChunk] = [
            EmbeddedChunk(chunk=c, embedding=_embed_text(c.text, dim)) for c in chunks
        ]
        # store meta for resolve
        for c in chunks:
            key = (str(c.doc_id), int(c.order))
            self._meta[key] = {
                "path": getattr(doc, "path", ""),
                "kind": "chunk",
                "text": c.text or "",
                "preview": (c.text or "")[:240],
            }
        self.index.upsert(embedded)
        return len(chunks)

    def index_path(self, path: str) -> int:
        """Parse and index a single path; return number of chunks stored."""
        for p in self.parsers:
            pp: Any = cast(Any, p)
            accepts = getattr(pp, "accepts", None)
            if callable(accepts) and accepts(path):
                doc: Document = pp.parse(path)
                return self.index_document(doc)
        # No parser accepted; nothing indexed.
        return 0

    # ---------- Search ----------

    def search_text(self, text: str, top_k: int = 5) -> List[Hit]:
        dim = getattr(self.index, 'dim', 64)
        eq = EmbeddedQuery(query=Query(text=text, top_k=top_k), embedding=_embed_text(text, dim))
        result = self.index.search(eq)
        return result.hits

    # ---------- Resolve ----------

    def resolve_meta(self, doc_id: DocId, chunk_id: ChunkId, order: int) -> Dict[str, Any]:
        key = (str(doc_id), int(order))
        return dict(self._meta.get(key, {}))

    # ---------- Observability ----------

    def get_stats(self) -> dict[str, object]:
        docs = {k[0] for k in self._meta.keys()}
        return {
            "total_documents": len(docs),
            "total_chunks": len(self._meta),
            "errors": [],
        }

    # ---------- Lifecycle ----------

    def shutdown(self) -> None:
        """Release resources gracefully (best-effort)."""
        idx = getattr(self, "index", None)
        for name in ("flush", "close", "shutdown"):
            fn = getattr(idx, name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
