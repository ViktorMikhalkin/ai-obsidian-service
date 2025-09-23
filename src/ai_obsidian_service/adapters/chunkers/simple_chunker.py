from __future__ import annotations

from collections.abc import Sequence
from ai_obsidian_service.core import Chunk, ChunkId, Chunker, DocId, Document

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> list[str]:
    res: list[str] = []
    n = len(text)
    step = max(max_chars - overlap, 1)
    i = 0
    while i < n:
        res.append(text[i : i + max_chars])
        i += step
    return res

class SimpleChunker(Chunker):
    def __init__(self, *, max_chars: int = 1000, overlap: int = 100) -> None:
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, doc: Document) -> Sequence[Chunk]:
        pieces: list[str] = chunk_text(doc.text or "", self.max_chars, self.overlap)
        chunks: list[Chunk] = []
        for idx, t in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=ChunkId(f"{doc.id}::{idx}"),
                    doc_id=DocId(doc.id),
                    order=idx,
                    text=t,
                )
            )
        return chunks
