from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

__all__ = ["Chunker", "SimpleChunker", "chunk_text"]

# Re-export canonical implementations from the concrete strategy module
from .simple_chunker import SimpleChunker, chunk_text  # noqa: F401


class Chunker(Protocol):
    """Minimal protocol for text chunkers.

    Implementations must provide:

        chunk(text: str) -> Sequence[Any]

    The returned items may be domain Chunk objects or plain strings.
    """
    def chunk(self, text: str) -> Sequence[Any]: ...  # pragma: no cover
