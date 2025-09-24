from __future__ import annotations

from dataclasses import dataclass

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.core import Chunker
from ai_obsidian_service.index.embedder_sentence_transformers import (
    SentenceTransformersEmbedder,
)
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.faiss_store import FaissVectorStore


@dataclass(slots=True)
class Components:
    embedder: SentenceTransformersEmbedder
    store: FaissVectorStore
    index: EmbeddingIndex
    parser: MarkdownParser
    search: SearchService


def make_components(*, chunker: Chunker, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Components:
    """Wire FAISS + SentenceTransformers with zero fallbacks."""
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    store = FaissVectorStore()
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
    search = SearchService(index=index, parser=parser)
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)
