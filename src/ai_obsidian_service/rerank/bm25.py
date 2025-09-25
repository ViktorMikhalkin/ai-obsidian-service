
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import log
from re import findall

from ai_obsidian_service.domain.models import Hit

_TOKEN_RE = r"\w+"

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in findall(_TOKEN_RE, text or "")]

def _hit_text(h: Hit) -> str:
    if h.chunk and _hit_text(h):
        return _hit_text(h)
    return h.snippet or ""

@dataclass(slots=True)
class BM25Reranker:
    k1: float = 1.5
    b: float = 0.75

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        if not hits or not query.strip():
            return list(hits)
        q_tokens = _tokenize(query)
        docs: list[list[str]] = [_tokenize(_hit_text(h)) for h in hits]
        N = max(1, len(docs))
        avgdl = sum(len(d) for d in docs) / N
        df: Counter[str] = Counter()
        for d in docs:
            df.update(set(d))
        idf: dict[str, float] = {}
        for t in set(q_tokens):
            n_t = df.get(t, 0)
            idf[t] = log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)

        def score_doc(tokens: list[str]) -> float:
            if not tokens:
                return 0.0
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for t in q_tokens:
                if t not in tf:
                    continue
                tf_t = tf[t]
                denom = tf_t + self.k1 * (1 - self.b + self.b * dl / max(1.0, avgdl))
                score += idf.get(t, 0.0) * (tf_t * (self.k1 + 1) / denom)
            return score

        scores = [score_doc(d) for d in docs]
        indexed = list(enumerate(hits))
        indexed.sort(key=lambda ih: (scores[ih[0]]), reverse=True)
        return [h for _, h in indexed]
