from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import DocId, Document
from ai_obsidian_service.adapters.chunkers import chunk_text


def test_simple_chunker_splits_text():
    text = "A" * 250
    doc = Document(id=DocId("doc-1"), path="/tmp/a.md", mime="text/markdown", text=text)
    ch = SimpleChunker(max_chars=100, overlap=10)
    chunks = ch.split(doc)
    assert len(chunks) >= 3
    assert chunks[0].text == "A" * 100
    assert chunks[1].text.startswith("A" * 10)  # overlap present


def test_chunker_basic():
    text = " ".join(str(i) for i in range(100))
    chunks = list(chunk_text(text, target_tokens=20, overlap_tokens=5))
    assert len(chunks) >= 4
    # Additional structure and bounds checks
    assert all(isinstance(c, tuple) and len(c) == 2 for c in chunks)
    assert all(isinstance(c[0], str) and c[0] for c in chunks)
    n = len(text)
    for _, (a, b) in chunks:
        assert 0 <= a <= b <= n
