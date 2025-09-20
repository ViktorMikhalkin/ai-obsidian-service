from collections.abc import Iterable, Sequence

from ai_obsidian_service.domain.models import (
    Chunk,
    ChunkId,
    DocId,
    Document,
    Hit,
    Query,
)
from ai_obsidian_service.ports.interfaces import (
    Chunker,
    DocumentParser,
    EmbeddingIndex,
    LlmClient,
)


class _FakeParser:
    def can_parse(self, path: str) -> bool:
        return path.endswith(".md")

    def parse(self, path: str) -> Document:
        return Document(id=DocId(path), path=path, mime="text/markdown", text="# Title")


class _FakeChunker:
    def split(self, doc):
        return [
            Chunk(
                id=ChunkId(f"{doc.id}_chunk_0"),  # Add the missing id field
                doc_id=doc.id,
                order=0,
                text=doc.text,
            )
        ]


class _FakeIndex:
    def upsert(self, chunks: Iterable[Chunk]) -> None:
        self._count = getattr(self, "_count", 0) + len(list(chunks))

    def search(self, query: Query) -> Sequence[Hit]:
        return [
            Hit(
                chunk_id=ChunkId("chunk-1"),  # Add this required field
                doc_id=DocId("doc-1"),
                chunk_order=0,
                score=0.9,
                snippet="...",
            )
        ]


class _FakeLlm:
    def generate(self, prompt: str) -> str:
        return f"echo: {prompt}"


def test_protocol_assignments_and_isinstance_checks():
    p: DocumentParser = _FakeParser()
    c: Chunker = _FakeChunker()
    idx: EmbeddingIndex = _FakeIndex()
    llm: LlmClient = _FakeLlm()

    assert p.can_parse("a.md")
    assert isinstance(p, DocumentParser)
    assert isinstance(c, Chunker)
    assert isinstance(idx, EmbeddingIndex)
    assert isinstance(llm, LlmClient)

    doc = p.parse("a.md")
    chunks = c.split(doc)
    idx.upsert(chunks)
    hits = idx.search(Query(text="hi", top_k=1))
    assert hits and hits[0].doc_id
