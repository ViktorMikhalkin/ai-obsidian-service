from __future__ import annotations
import os
from dataclasses import dataclass

from ai_obsidian_service.core import Chunker
from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
import os
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.rerank.bm25 import BM25Reranker
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index import Embedder, VectorStore

# Memory DI (из прошлой итерации)
from ai_obsidian_service.di import DummyEmbedder, InMemoryVectorStore  # без фоллбеков, это «боевой» memory
# FAISS + ST DI
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.index.embedder_sentence_transformers import SentenceTransformersEmbedder


@dataclass(slots=True)
class Components:
    embedder: Embedder
    store: VectorStore
    index: EmbeddingIndex
    parser: MarkdownParser
    search: SearchService


def _make_memory(*, chunker: Chunker) -> Components:
    embedder = DummyEmbedder()
    store = InMemoryVectorStore()
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
    enable_bm25 = os.getenv("AIOS_BM25", "1").lower() not in ("0", "false", "no")
rerank_topn = int(os.getenv("AIOS_RERANK_TOPN", "50"))
search = SearchService(index=index, parser=parser, reranker=(BM25Reranker() if enable_bm25 else None), rerank_topn=rerank_topn)
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)


def _make_faiss(*, chunker: Chunker, model_name: str) -> Components:
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    store = FaissVectorStore()
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
    enable_bm25 = os.getenv("AIOS_BM25", "1").lower() not in ("0", "false", "no")
rerank_topn = int(os.getenv("AIOS_RERANK_TOPN", "50"))
search = SearchService(index=index, parser=parser, reranker=(BM25Reranker() if enable_bm25 else None), rerank_topn=rerank_topn)
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)


def make_components(*, chunker: Chunker, backend: str | None = None, model_name: str | None = None) -> Components:
    """
    Pure selector with zero fallbacks:
    - backend: "memory" | "faiss"
    - model_name: sentence-transformers model (only for backend="faiss")

    If args are None, we read ENV:
      VECTOR_STORE_BACKEND = "memory" | "faiss"
      ST_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    """
    backend = backend or os.getenv("VECTOR_STORE_BACKEND", "memory").lower()
    if backend == "memory":
        return _make_memory(chunker=chunker)
    if backend == "faiss":
        model_name = model_name or os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        return _make_faiss(chunker=chunker, model_name=model_name)
    raise ValueError(f"Unsupported VECTOR_STORE_BACKEND={backend!r}")
