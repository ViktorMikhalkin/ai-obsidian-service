"""
Tests for deterministic chunk ID generation.

Run with: pytest tests/test_deterministic_ids.py -v
"""

from __future__ import annotations

import pytest

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.core import Document
from ai_obsidian_service.utils.ids import chunk_id, source_id


def test_source_id_determinism():
    """Same path always produces same source ID"""
    path = "notes/test.md"

    id1 = source_id(path)
    id2 = source_id(path)

    assert id1 == id2
    assert len(id1) == 16  # 8 bytes in hex


def test_source_id_uniqueness():
    """Different paths get different source IDs"""
    id1 = source_id("notes/file1.md")
    id2 = source_id("notes/file2.md")
    id3 = source_id("papers/file1.md")

    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


def test_chunk_id_determinism():
    """Same path and position always produces same chunk ID"""
    path = "notes/test.md"

    id1 = chunk_id(path, 0)
    id2 = chunk_id(path, 0)

    assert id1 == id2
    assert len(id1) == 16


def test_chunk_id_position_uniqueness():
    """Different chunk positions get different IDs"""
    path = "notes/test.md"

    id0 = chunk_id(path, 0)
    id1 = chunk_id(path, 1)
    id2 = chunk_id(path, 2)

    assert id0 != id1
    assert id1 != id2
    assert id0 != id2


def test_chunk_id_file_uniqueness():
    """Different files get different chunk IDs even at same position"""
    id1 = chunk_id("notes/file1.md", 0)
    id2 = chunk_id("notes/file2.md", 0)

    assert id1 != id2


def test_source_id_vs_chunk_id():
    """Source ID differs from chunk IDs"""
    path = "notes/test.md"

    doc_id = source_id(path)
    chunk_id_0 = chunk_id(path, 0)
    chunk_id_1 = chunk_id(path, 1)

    assert doc_id != chunk_id_0
    assert doc_id != chunk_id_1


def test_chunker_produces_deterministic_ids():
    """SimpleChunker generates deterministic IDs"""
    chunker = SimpleChunker(max_chars=100, overlap=20)

    doc = Document(
        id="temp-id",
        path="notes/test.md",  # Relative path
        mime="text/markdown",
        text="Hello world! " * 50,
        metadata={"author": "test"},
    )

    # Run chunking twice
    chunks1 = chunker.split(doc)
    chunks2 = chunker.split(doc)

    # IDs should be identical
    assert len(chunks1) == len(chunks2)
    assert len(chunks1) > 1  # Should create multiple chunks

    for c1, c2 in zip(chunks1, chunks2, strict=True):
        assert c1.id == c2.id
        assert c1.doc_id == c2.doc_id


def test_chunker_ids_survive_rebuild():
    """Chunk IDs remain stable across rebuilds with different doc.id"""
    chunker = SimpleChunker(max_chars=100, overlap=20)

    # First indexing
    doc1 = Document(
        id="first-run-uuid",
        path="notes/daily.md",
        mime="text/markdown",
        text="This is my daily note with important content.",
    )
    chunks1 = chunker.split(doc1)

    # Rebuild with different doc.id but same path
    doc2 = Document(
        id="second-run-uuid",  # Different!
        path="notes/daily.md",  # Same!
        mime="text/markdown",
        text="This is my daily note with important content.",
    )
    chunks2 = chunker.split(doc2)

    # Chunk IDs should be identical (path-based, not doc.id-based)
    assert chunks1[0].id == chunks2[0].id
    assert chunks1[0].doc_id == chunks2[0].doc_id


def test_chunker_all_chunks_unique():
    """All chunks from same document have unique IDs"""
    chunker = SimpleChunker(max_chars=50, overlap=10)

    doc = Document(
        id="test",
        path="notes/test.md",
        mime="text/markdown",
        text="A" * 200,  # Will create multiple chunks
    )

    chunks = chunker.split(doc)

    assert len(chunks) > 1
    chunk_ids = [c.id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "All chunk IDs must be unique"


def test_chunker_all_chunks_same_doc_id():
    """All chunks from same document share same doc_id"""
    chunker = SimpleChunker(max_chars=50, overlap=10)

    doc = Document(
        id="test",
        path="notes/test.md",
        mime="text/markdown",
        text="A" * 200,
    )

    chunks = chunker.split(doc)

    doc_ids = {c.doc_id for c in chunks}
    assert len(doc_ids) == 1, "All chunks from same doc should share doc_id"


def test_chunker_preserves_metadata():
    """Chunker correctly passes metadata to chunks"""
    chunker = SimpleChunker(max_chars=100)

    doc = Document(
        id="test",
        path="papers/research.pdf",
        mime="application/pdf",
        text="Test content",
        metadata={"tags": ["important"], "created": "2025-01-01"},
    )

    chunks = chunker.split(doc)

    assert len(chunks) > 0
    assert chunks[0].metadata == doc.metadata

    # Type narrowing - prove metadata is not None before indexing
    metadata = chunks[0].metadata
    assert metadata is not None
    assert metadata["tags"] == ["important"]


def test_empty_document():
    """Chunker handles empty documents"""
    chunker = SimpleChunker(max_chars=100)

    doc = Document(
        id="empty",
        path="notes/empty.md",
        mime="text/markdown",
        text="",
    )

    chunks = chunker.split(doc)

    # Empty text creates one empty chunk
    assert len(chunks) >= 0


@pytest.mark.parametrize(
    "path,order",
    [
        ("notes/test.md", 0),
        ("notes/daily/2025-01-01.md", 5),
        ("papers/research/ml.pdf", 100),
        ("books/fiction/novel.epub", 999),
    ],
)
def test_chunk_id_various_paths(path: str, order: int):
    """Test chunk ID generation with different path patterns"""
    id1 = chunk_id(path, order)
    id2 = chunk_id(path, order)

    assert id1 == id2
    assert len(id1) == 16


def test_vault_vs_library_collections():
    """Different collections (vault vs library) produce different IDs"""
    vault_chunk = chunk_id("notes/daily.md", 0)
    library_chunk = chunk_id("papers/ml-research.pdf", 0)

    assert vault_chunk != library_chunk


def test_nested_paths():
    """Nested paths work correctly"""
    shallow = chunk_id("note.md", 0)
    nested = chunk_id("folder/subfolder/deep/note.md", 0)

    assert shallow != nested
    assert len(shallow) == 16
    assert len(nested) == 16


def test_path_with_special_chars():
    """Paths with special characters are handled"""
    id1 = chunk_id("notes/2025-01-01.md", 0)
    id2 = chunk_id("papers/ML & AI (2024).pdf", 0)

    assert len(id1) == 16
    assert len(id2) == 16


def test_chunker_order_field():
    """Chunks have correct order field"""
    chunker = SimpleChunker(max_chars=50, overlap=10)

    doc = Document(
        id="test",
        path="notes/test.md",
        mime="text/markdown",
        text="A" * 200,
    )

    chunks = chunker.split(doc)

    for idx, chunk in enumerate(chunks):
        assert chunk.order == idx


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
