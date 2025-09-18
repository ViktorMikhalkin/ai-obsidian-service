import json
from pathlib import Path

from .embedder import Embedder
from .models import SearchHit
from .store.vector_faiss import FaissIndex

_dim_path = Path("index/dim.txt")
_meta_path = Path("index/index.jsonl")
_faiss_path = Path("index/faiss.index")

_dim = int(_dim_path.read_text()) if _dim_path.exists() else None
_meta = (
    [json.loads(load) for load in _meta_path.read_text().splitlines()]
    if _meta_path.exists()
    else []
)
_fa = FaissIndex(dim=_dim, path=_faiss_path) if _dim else None
_emb = Embedder() if _dim else None


def search(query: str, top_k: int = 8):
    if not (_fa and _meta and _emb):
        return []
    q = _emb.encode([query])[0]
    scores, idx = _fa.search(q, top_k=top_k)  # type: ignore[attr-defined]
    hits = []
    for rank, rid in enumerate(idx[0]):
        if rid < 0 or rid >= len(_meta):
            continue
        m = _meta[rid]
        s = float(scores[0][rank])
        hits.append(
            SearchHit(
                doc_path=m["path"], chunk_id=str(m["id"]), score=s, preview=m["preview"]
            )
        )
    return hits
