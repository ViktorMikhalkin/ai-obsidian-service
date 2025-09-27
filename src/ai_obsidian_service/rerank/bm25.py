from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import log
from typing import Any

from ai_obsidian_service.domain.models import Hit


def _tokenize(text: str) -> list[str]:
    # ultra-simple tokenizer: lowercase + split on non-alnum
    import re
    return [t for t in re.split(r"[^0-9a-zA-Z]+", (text or "").lower()) if t]


def _hit_text(h: Hit) -> str:
    # Prefer full chunk text
    chunk = getattr(h, "chunk", None)
    if chunk is not None:
        text = getattr(chunk, "text", None)
        if isinstance(text, str):
            return text

    # Fallback to snippet
    snippet = getattr(h, "snippet", None)
    if isinstance(snippet, str):
        return snippet

    # Fallback to metadata["path"]
    metadata: Any = getattr(h, "metadata", None)
    if metadata is None and chunk is not None:
        metadata = getattr(chunk, "metadata", None)

    if isinstance(metadata, dict):
        path = metadata.get("path")
        if isinstance(path, str):
            return path

    return ""


@dataclass(slots=True)
class BM25Reranker:
    """Lightweight BM25 re-ranker for top-N vector hits."""
    k1: float = 1.2
    b: float = 0.75

    def rerank(self, *, query: str, hits: Sequence[Hit], limit: int | None = None) -> list[Hit]:
        """
        BM25 re-ranking over retrieved hits.
        - Stable: ties keep original order.
        - Optional `limit`: truncate to top-N after scoring (used by SearchService).
        """
        if not hits:
            return []
        q = (query or "").strip()
        if not q:
            return list(hits)

        q_tokens = _tokenize(q)
        if not q_tokens:
            return list(hits)

        docs: list[list[str]] = [_tokenize(_hit_text(h)) for h in hits]
        N = max(1, len(docs))
        avgdl = sum(len(d) for d in docs) / float(N)

        # doc frequency only for query terms
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))

        idf: dict[str, float] = {}
        for t in set(q_tokens):
            n_t = df.get(t, 0)
            # +1 smoothing to avoid negative surprises on very common terms
            idf[t] = log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)

        k1 = float(self.k1)
        b = float(self.b)

        def score_doc(tokens: list[str]) -> float:
            if not tokens:
                return 0.0
            tf = Counter(tokens)
            dl = float(len(tokens))
            denom_norm = k1 * (1.0 - b + b * (dl / max(1.0, avgdl)))
            s = 0.0
            for t in q_tokens:
                tf_t = tf.get(t)
                if not tf_t:
                    continue
                denom = tf_t + denom_norm
                s += idf.get(t, 0.0) * (tf_t * (k1 + 1.0) / denom)
            return s

        scores = [score_doc(d) for d in docs]

        # stable sort by (-score, original_index)
        indexed = list(enumerate(hits))
        indexed.sort(key=lambda ih: (-scores[ih[0]], ih[0]))
        ranked = [h for _, h in indexed]

        if limit is not None and limit >= 0:
            return ranked[:limit]
        return ranked
