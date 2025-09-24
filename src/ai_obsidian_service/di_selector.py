from __future__ import annotations

import os
from dataclasses import dataclass

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser
from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.di import DummyEmbedder, InMemoryVectorStore
from ai_obsidian_service.index import Embedder, VectorStore
from ai_obsidian_service.index.embedder_sentence_transformers import (
    SentenceTransformersEmbedder,
)
from ai_obsidian_service.index.embedding_index import EmbeddingIndex
from ai_obsidian_service.index.faiss_store import FaissVectorStore
from ai_obsidian_service.rerank.bm25 import BM25Reranker


class Chunker:  # protocol-like shim
    def split(self, doc):  # type: ignore[no-untyped-def]
        raise NotImplementedError

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
    search = SearchService(
        index=index,
        parser=parser,
        reranker=(BM25Reranker() if enable_bm25 else None),
        rerank_topn=rerank_topn,
    )
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)

def _make_faiss(*, chunker: Chunker, model_name: str) -> Components:
    embedder = SentenceTransformersEmbedder(model_name=model_name)
    store = FaissVectorStore()
    parser = MarkdownParser()
    index = EmbeddingIndex(embedder=embedder, store=store, chunker=chunker)
    enable_bm25 = os.getenv("AIOS_BM25", "1").lower() not in ("0", "false", "no")
    rerank_topn = int(os.getenv("AIOS_RERANK_TOPN", "50"))
    search = SearchService(
        index=index,
        parser=parser,
        reranker=(BM25Reranker() if enable_bm25 else None),
        rerank_topn=rerank_topn,
    )
    return Components(embedder=embedder, store=store, index=index, parser=parser, search=search)

def make_components(*, chunker: Chunker, backend: str | None = None, model_name: str | None = None) -> Components:
    """
    Pure selector:
    - backend: "memory" | "faiss"
    - model_name: ST model if backend="faiss"
    """
    _backend = (backend or os.getenv("VECTOR_STORE_BACKEND", "memory")).lower()
    if _backend == "memory":
        return _make_memory(chunker=chunker)
    if _backend == "faiss":
        _model_name = model_name or os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        return _make_faiss(chunker=chunker, model_name=_model_name)
    raise ValueError(f"Unsupported VECTOR_STORE_BACKEND={_backend!r}")
