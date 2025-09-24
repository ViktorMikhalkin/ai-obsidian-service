import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # skip if FAISS not installed

from ai_obsidian_service.core import Chunk, Chunker, Document
from ai_obsidian_service.di_faiss import make_components


class FixedLineChunker(Chunker):
    """Very simple chunker: one non-empty line -> one chunk."""
    def split(self, doc: Document):
        for i, line in enumerate(doc.text.splitlines()):
            t = line.strip()
            if t:
                yield Chunk(id=f"{doc.id}#{i}", text=t, meta={"line": i})


def test_di_faiss_end_to_end(tmp_path):
    # DI wiring: SentenceTransformers + FaissVectorStore + MarkdownParser
    comps = make_components(chunker=FixedLineChunker())

    # Prepare a tiny document
    p = tmp_path / "doc.md"
    p.write_text("# H\nalpha\nbeta\n", encoding="utf-8")
    doc = comps.parser.parse(str(p))

    # Index it through the facade (EmbeddingIndex)
    n = comps.index.index_document(doc)
    assert n == 2  # only "alpha" and "beta" (header line is not a chunk)

    # Search through SearchService (does not know about embedding)
    result = comps.search.search_text("alpha", top_k=1)
    assert len(result.hits) == 1
    assert result.hits[0].chunk.text in {"alpha", "beta"}  # depending on model similarity

    # sanity: vectors are normalized in embedder; FAISS IP ~ cosine
    vec = comps.embedder.embed("alpha")
    assert vec.dtype == np.float32
    # close to 1.0 within tolerance
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5
