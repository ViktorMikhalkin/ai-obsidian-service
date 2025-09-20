from collections.abc import Iterable

from ai_obsidian_service.domain.models import (
    Chunk,
    ChunkId,
    DocId,
    Document,
    EmbeddedChunk,
    EmbeddedQuery,
    Hit,
    Query,
    SearchResult,
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
    def upsert(self, embedded_chunks: Iterable[EmbeddedChunk]) -> None:
        self._count = getattr(self, "_count", 0) + len(list(embedded_chunks))

    def search(self, embedded_query: EmbeddedQuery) -> SearchResult:
        hits = [
            Hit(
                chunk_id=ChunkId("chunk-1"),
                doc_id=DocId("doc-1"),
                chunk_order=0,
                score=0.9,
                snippet="...",
            )
        ]
        return SearchResult(
            query=embedded_query.query,
            hits=hits,
            total_time_ms=1.0,
            retrieved_at="2025-01-01T00:00:00",
        )


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

    # Create embedded chunks for the EmbeddingIndex
    import numpy as np

    embedded_chunks = [
        EmbeddedChunk(chunk=chunk, embedding=np.array([0.1, 0.2, 0.3]))
        for chunk in chunks
    ]

    idx.upsert(embedded_chunks)

    # Create embedded query for the search
    embedded_query = EmbeddedQuery(
        query=Query(text="hi", top_k=1), embedding=np.array([0.1, 0.2, 0.3])
    )

    result = idx.search(embedded_query)
    assert result.hits and result.hits[0].doc_id
