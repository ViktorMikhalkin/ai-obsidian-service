from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from ai_obsidian_service.adapters.parsers.epub_parser import EpubParser
from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.parsers.pdf_parser import PdfParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Chunker
from ai_obsidian_service.index.embedder_sentence_transformers import (
    SentenceTransformersEmbedder,
)
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.ports.interfaces import DocumentParser


@dataclass(slots=True)
class Components:
    embedder: SentenceTransformersEmbedder
    store: FaissVectorStore
    index: EmbeddingIndex
    parsers: Sequence[DocumentParser]
    search: SearchService


def make_components(
    *, chunker: Chunker, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Components:
    """Wire FAISS + SentenceTransformers with zero fallbacks."""
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    store = FaissVectorStore()
    raw = cast(
        Sequence[DocumentParser],
        cast(object, [MarkdownParser(), PdfParser(), EpubParser()]),
    )
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
    search = SearchService(index=index, parsers=raw)
    return Components(
        embedder=embedder, store=store, index=index, parsers=raw, search=search
    )
