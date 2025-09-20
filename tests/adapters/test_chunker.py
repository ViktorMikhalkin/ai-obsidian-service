from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import DocId, Document


def test_simple_chunker_splits_text():
    text = "A" * 250
    doc = Document(id=DocId("doc-1"), path="/tmp/a.md", mime="text/markdown", text=text)
    ch = SimpleChunker(max_chars=100, overlap=10)
    chunks = ch.split(doc)
    assert len(chunks) >= 3
    assert chunks[0].text == "A" * 100
    assert chunks[1].text.startswith("A" * 10)  # overlap present
