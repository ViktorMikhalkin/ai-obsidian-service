from ai_obsidian_service.core import Chunk, Chunker, ChunkId, DocId, Document


class SimpleChunker(Chunker):
    """Character-based chunker with optional overlap."""

    def __init__(self, max_chars: int = 1000, overlap: int = 100) -> None:
        assert max_chars > 0, "max_chars must be positive"
        assert overlap >= 0, "overlap must be non-negative"
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text or ""
        if not text:
            return [
                Chunk(id=ChunkId(f"{doc.id}#0"), doc_id=DocId(doc.id), order=0, text="")
            ]
        chunks: list[Chunk] = []
        start = 0
        order = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            segment = text[start:end]
            chunks.append(
                Chunk(
                    id=ChunkId(f"{doc.id}#{order}"),
                    doc_id=DocId(doc.id),
                    order=order,
                    text=segment,
                )
            )
            if end == len(text):
                break
            next_start = end - self.overlap
            start = next_start if next_start > start else end
            order += 1
        return chunks

def chunk_text(text: str, max_chars: int = 1000) -> list[str]:
    """Shortcut compatible with legacy tests."""
    return SimpleChunker(max_chars=max_chars).chunk(text)
