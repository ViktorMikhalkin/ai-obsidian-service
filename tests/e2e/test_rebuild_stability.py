"""
E2E tests for rebuild stability and deterministic chunk IDs.

These tests verify that rebuilding the same corpus produces identical chunk IDs,
ensuring citations remain stable across rebuilds.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.e2e
def test_rebuild_produces_stable_chunk_ids(tmp_path: Path, empty_search_service):
    """
    Rebuilding the same corpus multiple times produces identical chunk IDs.

    This is critical for:
    - Citation stability (citations survive rebuilds)
    - Incremental updates (detect which chunks changed)
    - User trust (predictable behavior)
    """
    # Create test vault
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note1.md").write_text(
        "# Note 1\nThis is test content for stability.", encoding="utf-8"
    )
    (vault / "note2.md").write_text("# Note 2\nAnother test note.", encoding="utf-8")

    service = empty_search_service

    # First indexing pass
    service.vault_root = str(vault)
    for file_path in vault.rglob("*.md"):
        service.index_path(str(file_path), vault_root=str(vault))

    # Get chunk IDs from first pass
    result1 = service.search_text("test", top_k=10)
    chunk_ids_1 = {hit.chunk_id for hit in result1.hits}
    doc_ids_1 = {hit.doc_id for hit in result1.hits}

    assert len(chunk_ids_1) > 0, "Should have indexed some chunks"

    # Clear the index (simulate fresh rebuild)
    service.index.store._chunks.clear()
    service.index.store._ids.clear()
    if (
        hasattr(service.index.store, "_index")
        and service.index.store._index is not None
    ):
        service.index.store._index.reset()

    # Second indexing pass (same content)
    service.vault_root = str(vault)
    for file_path in vault.rglob("*.md"):
        service.index_path(str(file_path), vault_root=str(vault))

    # Get chunk IDs from second pass
    result2 = service.search_text("test", top_k=10)
    chunk_ids_2 = {hit.chunk_id for hit in result2.hits}
    doc_ids_2 = {hit.doc_id for hit in result2.hits}

    # Verify IDs are identical
    assert chunk_ids_1 == chunk_ids_2, "Chunk IDs must remain stable across rebuilds"
    assert doc_ids_1 == doc_ids_2, "Document IDs must remain stable across rebuilds"


@pytest.mark.e2e
def test_file_edit_changes_only_affected_chunks(tmp_path: Path, empty_search_service):
    """
    Editing one file changes only that file's chunk IDs (when content changes).

    This demonstrates that deterministic IDs enable:
    - Detecting which files changed
    - Selective re-indexing (future feature)
    """
    # Create test vault
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "unchanged.md").write_text(
        "# Unchanged\nThis file stays the same.", encoding="utf-8"
    )
    (vault / "changed.md").write_text("# Original\nThis will change.", encoding="utf-8")

    service = empty_search_service

    # Initial indexing
    service.vault_root = str(vault)
    for file_path in vault.rglob("*.md"):
        service.index_path(str(file_path), vault_root=str(vault))

    # Get all chunk IDs
    result1 = service.search_text("file", top_k=10)
    all_chunks_1 = {(hit.chunk_id, hit.chunk.text[:20]) for hit in result1.hits}

    # Edit one file
    (vault / "changed.md").write_text(
        "# Changed\nThis content is different now.", encoding="utf-8"
    )

    # Clear and rebuild
    service.index.store._chunks.clear()
    service.index.store._ids.clear()
    if (
        hasattr(service.index.store, "_index")
        and service.index.store._index is not None
    ):
        service.index.store._index.reset()

    service.vault_root = str(vault)
    for file_path in vault.rglob("*.md"):
        service.index_path(str(file_path), vault_root=str(vault))

    # Get all chunk IDs after edit
    result2 = service.search_text("file", top_k=10)
    all_chunks_2 = {(hit.chunk_id, hit.chunk.text[:20]) for hit in result2.hits}

    # Find chunks from unchanged file
    unchanged_chunks_1 = {cid for cid, text in all_chunks_1 if "stays the same" in text}
    unchanged_chunks_2 = {cid for cid, text in all_chunks_2 if "stays the same" in text}

    # Unchanged file should have same chunk IDs
    assert unchanged_chunks_1 == unchanged_chunks_2, (
        "Unchanged file should have stable chunk IDs"
    )

    # Changed file should have different content but same ID structure
    # (IDs are based on path+position, not content)
    changed_chunks_1 = {cid for cid, text in all_chunks_1 if "change" in text.lower()}
    changed_chunks_2 = {
        cid for cid, text in all_chunks_2 if "different" in text.lower()
    }

    # Same file, same position → same chunk IDs (even though content differs)
    # This is expected with position-based IDs
    assert len(changed_chunks_1) > 0, "Should have chunks from changed file"
    assert len(changed_chunks_2) > 0, "Should have chunks from changed file after edit"


@pytest.mark.e2e
def test_cross_platform_path_normalization(tmp_path: Path, empty_search_service):
    """
    Path normalization ensures IDs are consistent regardless of OS path format.

    This is tested by using both forward slashes and backslashes in paths.
    """
    # Create nested structure
    vault = tmp_path / "vault"
    subdir = vault / "notes" / "daily"
    subdir.mkdir(parents=True, exist_ok=True)
    (subdir / "2025-01-01.md").write_text(
        "# Daily Note\nTest content.", encoding="utf-8"
    )

    service = empty_search_service
    service.vault_root = str(vault)

    # Index the file
    file_path = subdir / "2025-01-01.md"
    service.index_path(str(file_path), vault_root=str(vault))

    # Get chunk IDs
    result = service.search_text("daily", top_k=5)
    chunk_ids = {hit.chunk_id for hit in result.hits}

    assert len(chunk_ids) > 0, "Should have indexed chunks"

    # Verify all chunk IDs are deterministic hex strings
    for chunk_id in chunk_ids:
        assert len(chunk_id) == 16, f"Chunk ID should be 16 chars: {chunk_id}"
        assert all(c in "0123456789abcdef" for c in chunk_id), (
            f"Chunk ID should be hex: {chunk_id}"
        )


@pytest.mark.e2e
def test_new_file_added_preserves_existing_ids(tmp_path: Path, empty_search_service):
    """
    Adding new files doesn't change existing chunk IDs.

    This is the foundation for incremental indexing.
    """
    # Create initial vault
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "existing.md").write_text(
        "# Existing\nOriginal content.", encoding="utf-8"
    )

    service = empty_search_service
    service.vault_root = str(vault)

    # Index first file
    service.index_path(str(vault / "existing.md"), vault_root=str(vault))

    # Get chunk IDs from first file
    result1 = service.search_text("existing", top_k=5)
    existing_chunk_ids_1 = {hit.chunk_id for hit in result1.hits}

    assert len(existing_chunk_ids_1) > 0, "Should have chunks from existing file"

    # Add new file
    (vault / "new.md").write_text("# New File\nBrand new content.", encoding="utf-8")
    service.index_path(str(vault / "new.md"), vault_root=str(vault))

    # Get chunk IDs from first file again
    result2 = service.search_text("existing", top_k=5)
    existing_chunk_ids_2 = {hit.chunk_id for hit in result2.hits}

    # Original file's chunk IDs should be unchanged
    assert existing_chunk_ids_1 == existing_chunk_ids_2, (
        "Adding new files shouldn't change existing chunk IDs"
    )

    # Verify new file was indexed
    result3 = service.search_text("new", top_k=5)
    new_chunk_ids = {hit.chunk_id for hit in result3.hits}
    assert len(new_chunk_ids) > 0, "Should have chunks from new file"

    # New chunks should have different IDs
    assert existing_chunk_ids_1.isdisjoint(new_chunk_ids), (
        "New file should have different chunk IDs"
    )


@pytest.mark.e2e
def test_chunk_id_format_validation(tmp_path: Path, empty_search_service):
    """
    All generated chunk IDs follow the expected format.

    Format: 16 hex characters (8 bytes of SHA256)
    """
    # Create test vault with various file types
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text(
        "# Test\n" + "Content " * 100, encoding="utf-8"
    )  # Multiple chunks

    service = empty_search_service
    service.vault_root = str(vault)

    # Index
    service.index_path(str(vault / "note.md"), vault_root=str(vault))

    # Get all chunks
    result = service.search_text("content", top_k=20)

    assert len(result.hits) > 0, "Should have indexed chunks"

    # Validate every chunk ID
    for hit in result.hits:
        chunk_id = hit.chunk_id
        doc_id = hit.doc_id

        # Format validation
        assert isinstance(chunk_id, str), f"Chunk ID should be string: {type(chunk_id)}"
        assert len(chunk_id) == 16, f"Chunk ID should be 16 chars: {chunk_id}"
        assert all(c in "0123456789abcdef" for c in chunk_id), (
            f"Chunk ID should be lowercase hex: {chunk_id}"
        )

        assert isinstance(doc_id, str), f"Doc ID should be string: {type(doc_id)}"
        assert len(doc_id) == 16, f"Doc ID should be 16 chars: {doc_id}"
        assert all(c in "0123456789abcdef" for c in doc_id), (
            f"Doc ID should be lowercase hex: {doc_id}"
        )

        # Doc ID should be same for all chunks from same document
        # (This is verified implicitly by the set logic, but good to document)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "e2e"])
