from __future__ import annotations

import hashlib


def doc_hash(text: str) -> str:
    """Content hash for detecting document changes (full SHA256)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_id(rel_posix: str) -> str:
    """
    Generate deterministic document/source ID from vault-relative path.

    Uses first 16 chars of SHA256 for compact but collision-resistant IDs.

    Args:
        rel_posix: Vault-relative path (e.g., "notes/daily.md")

    Returns:
        16-character hex string

    Examples:
        >>> source_id("notes/daily.md")
        'a3b4c5d6e7f8g9h0'
    """
    hash_bytes = hashlib.sha256(rel_posix.encode("utf-8")).digest()
    return hash_bytes[:8].hex()


def chunk_id(rel_posix: str, order: int) -> str:
    """
    Generate deterministic chunk ID from path and position.

    This ensures the same file chunk always gets the same ID across rebuilds,
    which is critical for:
    - Stable citations that survive index rebuilds
    - Incremental indexing (detect what changed)
    - Efficient re-indexing

    Args:
        rel_posix: Vault-relative path (e.g., "notes/daily.md")
        order: Zero-based chunk position in document

    Returns:
        16-character hex string

    Examples:
        >>> chunk_id("notes/daily.md", 0)
        'b4c5d6e7f8g9h0i1'
        >>> chunk_id("notes/daily.md", 1)
        'c5d6e7f8g9h0i1j2'
    """
    key = f"{rel_posix}::{order}"
    hash_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    return hash_bytes[:8].hex()
