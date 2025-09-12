from indexer.parsers.chunker import chunk_text


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
