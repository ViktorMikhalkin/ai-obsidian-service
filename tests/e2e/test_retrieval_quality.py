from __future__ import annotations

from math import log2

import pytest

from ai_obsidian_service.config.container import build_search_service

# ---- tiny golden truth (doc path, chunk_order=0)
GOLD = {
    "alpha": {("notes/A.md", 0)},
    "bravo": {("notes/A.md", 0), ("notes/B.md", 0)},
    "delta": {("notes/B.md", 0), ("docs/C.md", 0)},
}

def precision_at_k(pred: list[tuple[str,int]], truth: set[tuple[str,int]], k: int) -> float:
    got = pred[:k]
    hit = sum(1 for x in got if x in truth)
    return hit / max(1, k)

def dcg_at_k(rel: list[int], k: int) -> float:
    return sum(rel[i] / log2(i+2) for i in range(min(k, len(rel))))

def ndcg_at_k(pred: list[tuple[str,int]], truth: set[tuple[str,int]], k: int) -> float:
    rel = [1 if x in truth else 0 for x in pred[:k]]
    idcg = dcg_at_k(sorted(rel, reverse=True), k)
    return (dcg_at_k(rel, k) / idcg) if idcg > 0 else 0.0

@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory):
    # Build a real service with markdown parser + simple chunker; memory vector store per env
    s = build_search_service(index_dir=None)

    # Note: SearchService has 'parsers' (list), not 'parser' (single)
    # and chunker is part of the index, not the service directly
    # The service is already configured with appropriate parsers and chunker
    # from build_search_service, so we don't need to override them

    # create mini vault
    root = tmp_path_factory.mktemp("vault")
    (root / "notes").mkdir(parents=True, exist_ok=True)
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "notes" / "A.md").write_text("alpha bravo charlie\n", encoding="utf-8")
    (root / "notes" / "B.md").write_text("bravo delta\n", encoding="utf-8")
    (root / "docs" / "C.md").write_text("delta echo\n", encoding="utf-8")

    # index all three
    for p in (root / "notes").glob("*.md"):
        s.index_path(str(p))
    for p in (root / "docs").glob("*.md"):
        s.index_path(str(p))

    # stash root for path checks
    s._gold_root = str(root)  # type: ignore[attr-defined]
    return s

def _to_pred_list(hits):
    out = []
    for h in hits:
        # resolve doc path & order from meta if available
        path, order = None, getattr(h, "chunk_order", 0)
        try:
            if h.chunk and getattr(h.chunk, "meta", None):
                path = h.chunk.meta.get("path")
        except Exception:
            pass
        if path is None:
            path = str(h.doc_id)
        out.append((path, int(order or 0)))
    return out

@pytest.mark.e2e
def test_retrieval_precision_and_ndcg(service):
    THRESH_P5 = 0.6   # tune thresholds to avoid flakiness across embeddings
    THRESH_N5 = 0.6
    K = 5

    for q, truth in GOLD.items():
        res = service.search_text(q, top_k=K)
        pred = _to_pred_list(res.hits)

        p5 = precision_at_k(pred, truth, K)
        n5 = ndcg_at_k(pred, truth, K)
        assert p5 >= THRESH_P5, f"{q}: P@{K}={p5:.2f} < {THRESH_P5}"
        assert n5 >= THRESH_N5, f"{q}: nDCG@{K}={n5:.2f} < {THRESH_N5}"

@pytest.mark.e2e
def test_collection_filter_limits_results(service):
    # Ask "delta" but restrict to notes → should NOT return docs/C.md
    res = service.search_text("delta", top_k=5, collection="notes")
    pred = _to_pred_list(res.hits)
    assert all(path.startswith("notes/") for path, _ in pred), pred
