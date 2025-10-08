from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from ai_obsidian_service.core import Chunk, Chunker, ChunkId, DocId, Document


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


class SemanticChunker(Chunker):
    """
    Token-aware semantic chunker with overlap.

    Improvements over SimpleChunker:
    - Uses tiktoken for accurate token counting (critical for LLMs)
    - Configurable chunk size in tokens (not chars)
    - Overlap in tokens for better context preservation
    - Falls back to char-based if tiktoken unavailable

    Args:
        max_tokens: Target chunk size in tokens (500-1000 recommended)
        overlap_tokens: Overlap size in tokens (50-150 recommended)
        encoding: tiktoken encoding name (default: cl100k_base for GPT-4/OpenAI)
    """

    def __init__(
        self,
        *,
        max_tokens: int = 750,
        overlap_tokens: int = 75,
        encoding: str = "cl100k_base",
    ) -> None:
        self.max_tokens = int(max_tokens)
        self.overlap_tokens = int(overlap_tokens)
        self.encoding_name = encoding

        # Try to load tiktoken for accurate token counting
        self._encoder = None
        try:
            import tiktoken

            self._encoder = tiktoken.get_encoding(encoding)
            self._use_tokens = True
        except (ImportError, Exception):
            # Fallback to char-based (approximate)
            self._use_tokens = False
            # Rough approximation: 1 token ≈ 4 chars
            self._char_multiplier = 4

    def _chunk_by_tokens(self, text: str) -> list[str]:
        """Chunk text using tiktoken (accurate)."""
        if not self._encoder:
            return self._chunk_by_chars(text)

        tokens = self._encoder.encode(text)
        chunks = []

        start = 0
        step = max(self.max_tokens - self.overlap_tokens, 1)

        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self._encoder.decode(chunk_tokens)

            if chunk_text.strip():
                chunks.append(chunk_text)

            start += step

            # Avoid infinite loop
            if start >= len(tokens):
                break

        return chunks

    def _chunk_by_chars(self, text: str) -> list[str]:
        """Fallback: chunk by characters (approximate tokens)."""
        max_chars = self.max_tokens * self._char_multiplier
        overlap_chars = self.overlap_tokens * self._char_multiplier

        chunks = []
        n = len(text)
        step = max(max_chars - overlap_chars, 1)

        i = 0
        while i < n:
            chunk = text[i : i + max_chars]
            if chunk.strip():
                chunks.append(chunk)
            i += step

        return chunks

    def split(self, doc: Document) -> Sequence[Chunk]:
        """
        Split document into semantic chunks.

        Args:
            doc: Document to chunk

        Returns:
            List of Chunk objects with deterministic IDs
        """
        text = doc.text or ""
        if not text.strip():
            return []

        # Choose chunking strategy
        if self._use_tokens:
            pieces = self._chunk_by_tokens(text)
        else:
            pieces = self._chunk_by_chars(text)

        # Generate deterministic IDs
        source_path = doc.path if hasattr(doc, "path") and doc.path else str(doc.id)
        deterministic_doc_id = generate_deterministic_id(source_path)

        chunks: list[Chunk] = []
        for idx, chunk_text in enumerate(pieces):
            chunk_id = generate_deterministic_id(source_path, idx)

            chunks.append(
                Chunk(
                    id=ChunkId(chunk_id),
                    doc_id=DocId(deterministic_doc_id),
                    order=idx,
                    text=chunk_text,
                    metadata={
                        **(doc.metadata or {}),
                        # Add chunk-specific metadata
                        "chunk_index": idx,
                        "total_chunks": len(pieces),
                        "chunking_method": "tokens" if self._use_tokens else "chars",
                    },
                )
            )

        return chunks


# Backward compatibility alias
class SimpleChunker(SemanticChunker):
    """
    Backward compatible wrapper for SemanticChunker.

    Converts old char-based parameters to token-based equivalents.
    """

    def __init__(self, *, max_chars: int = 1000, overlap: int = 100) -> None:
        # Convert chars to approximate tokens (1 token ≈ 4 chars)
        max_tokens = max_chars // 4
        overlap_tokens = overlap // 4

        super().__init__(
            max_tokens=max(max_tokens, 100),  # minimum 100 tokens
            overlap_tokens=max(overlap_tokens, 10),  # minimum 10 tokens overlap
        )
