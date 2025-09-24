import pytest

try:
    from hypothesis import given  # type: ignore
    from hypothesis import strategies as st
except Exception:  # pragma: no cover
    pytest.skip("hypothesis not installed", allow_module_level=True)

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import DocId, Document


@given(st.text(min_size=1, max_size=2000), st.integers(min_value=50, max_value=400), st.integers(min_value=0, max_value=50))
def test_chunk_boundaries_monotonic(s: str, max_chars: int, overlap: int) -> None:
    ch = SimpleChunker(max_chars=max_chars, overlap=overlap)
    doc = Document(id=DocId("X"), path="X", mime="text/markdown", text=s, metadata={})
    chunks = list(ch.split(doc))
    assert all(chunks[i].order == i for i in range(len(chunks)))
    assert all((c.text or "") != "" for c in chunks)
