import pytest

try:
    pass  # type: ignore
except Exception:  # pragma: no cover
    pytest.skip("faiss is not installed", allow_module_level=True)

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.di_selector import make_components


def test_di_faiss_smoke(tmp_path):
    comps = make_components(chunker=SimpleChunker(), backend="faiss", model_name="sentence-transformers/all-MiniLM-L6-v2")
    # skip if components are not faiss-backed for any reason
    if comps.store.__class__.__name__.lower().find("faiss") < 0:
        pytest.skip("FaissVectorStore not active")

    # quick spherical normalization test for the embedder
    import numpy as _np
    vec = comps.index.embedder.embed("hello")  # type: ignore[attr-defined]
    assert vec.ndim == 1 and vec.dtype == _np.float32
    n = _np.linalg.norm(vec)
    assert 0.0 < n < 10.0
