from indexer.parsers.chunker import chunk_text

def test_chunker_basic():
    text = " ".join(str(i) for i in range(100))
    chunks = list(chunk_text(text, target_tokens=20, overlap_tokens=5))
    assert len(chunks) >= 4
