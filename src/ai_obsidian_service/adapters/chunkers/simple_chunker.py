from ai_obsidian_service.core import Chunk, Chunker, DocId, Document


class SimpleChunker(Chunker):
    """Character-based chunker with optional overlap. SRP-friendly, deterministic."""

    def __init__(self, max_chars: int = 1000, overlap: int = 100) -> None:
        assert max_chars > 0, "max_chars must be positive"
        assert overlap >= 0, "overlap must be non-negative"
        self.max_chars = max_chars
        self.overlap = overlap

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text or ""
        if not text:
            return [Chunk(doc_id=DocId(doc.id), order=0, text="")]
        chunks: list[Chunk] = []
        start = 0
        order = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            segment = text[start:end]
            chunks.append(Chunk(doc_id=DocId(doc.id), order=order, text=segment))
            if end == len(text):
                break
            # move start forward with overlap
            next_start = end - self.overlap
            start = next_start if next_start > start else end
            order += 1
        return chunks
