from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from ai_obsidian_service.core import Chunk, Chunker, ChunkId, DocId, Document


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    res: list[str] = []
    n = len(text)
    step = max(max_chars - overlap, 1)
    i = 0
    while i < n:
        res.append(text[i : i + max_chars])
        i += step
    return res


def generate_deterministic_id(source_path: str, chunk_index: int | None = None) -> str:
    """
    Generate deterministic ID based on normalized path and optional chunk index.

    This ensures the same file/chunk always gets the same ID across rebuilds.

    Args:
        source_path: Path to source document
        chunk_index: Optional chunk position (None for document ID)

    Returns:
        16-character hex string (deterministic)
    """
    # Normalize path for cross-OS consistency
    normalized = str(Path(source_path).as_posix()).lower()

    # Create key
    if chunk_index is not None:
        key = f"{normalized}::{chunk_index}"
    else:
        key = normalized

    # Generate deterministic ID
    hash_bytes = hashlib.sha256(key.encode("utf-8")).digest()
    return hash_bytes[:8].hex()


class SimpleChunker(Chunker):
    def __init__(self, *, max_chars: int = 1000, overlap: int = 100) -> None:
        self.max_chars = int(max_chars)
        self.overlap = int(overlap)

    def split(self, doc: Document) -> Sequence[Chunk]:
        pieces: list[str] = chunk_text(doc.text or "", self.max_chars, self.overlap)
        chunks: list[Chunk] = []

        # Generate deterministic doc_id from path
        # Use doc.path if available, fallback to doc.id
        source_path = doc.path if hasattr(doc, "path") and doc.path else str(doc.id)
        deterministic_doc_id = generate_deterministic_id(source_path)

        for idx, t in enumerate(pieces):
            # Generate deterministic chunk_id from path + position
            chunk_id = generate_deterministic_id(source_path, idx)

            chunks.append(
                Chunk(
                    id=ChunkId(chunk_id),
                    doc_id=DocId(deterministic_doc_id),
                    order=idx,
                    text=t,
                    metadata=doc.metadata,
                )
            )
        return chunks
