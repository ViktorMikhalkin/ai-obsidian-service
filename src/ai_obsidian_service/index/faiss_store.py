from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from ai_obsidian_service.domain.models import EmbeddedChunk, Hit, SearchResult
from ai_obsidian_service.index.vector_store import VectorStore


# LAZY IMPORT: Only import faiss when actually needed
def _get_faiss():
    """Lazy import of faiss to avoid segfault on module import."""
    try:
        import faiss
        return faiss
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "FAISS is required for FaissVectorStore. Install `faiss-cpu` (or `faiss-gpu`)."
        ) from e


def _read_json(p: str | Path) -> dict[str, Any]:
    # Cast for mypy
    result = json.loads(Path(p).read_text(encoding="utf-8"))
    return cast(dict[str, Any], result)


def _default_tmp_index_dir() -> str:
    root = os.path.join(tempfile.gettempdir(), "ai-obsidian-service", "faiss")
    return root


def _ensure_dir(p: str | Path) -> None:
    auto = os.getenv("AIOS_INDEX_AUTO_CREATE", "1").lower() not in ("0", "false", "no")
    if not auto:
        if not Path(p).exists():
            raise RuntimeError(f"Index directory does not exist: {p}")
        return
    try:
        Path(p).mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        raise RuntimeError(f"Cannot create index_dir {p}: {e}") from e


@dataclass(slots=True)
class FaissVectorStore(VectorStore):
    """FAISS-backed VectorStore using inner-product (cosine when vectors are L2-normalized).

    Persistence:
      - save(dir): writes
          dir/index.faiss        — FAISS index (fast-load)
          dir/embeddings.npy     — float32 [N, D] embeddings (source of truth for updates)
          dir/meta.json          — {version, dim, count, ids[], model_name?}
          dir/chunks.jsonl       — {"id", "text", "meta"} per line (for results)
      - load(dir): validates meta and rebuilds in-memory structures.

    Notes:
      - We keep BOTH the faiss index and embeddings.npy. If index load fails or dims mismatch,
        we rebuild FAISS from embeddings.npy.
      - Updates rely on embeddings in memory; hence we persist embeddings.npy.
    """

    dim: int | None = None
    _index: Any = field(default=None, init=False, repr=False)  # faiss.Index | None
    _ids: list[str] = field(default_factory=list, init=False, repr=False)
    _chunks: dict[str, EmbeddedChunk] = field(default_factory=dict, init=False, repr=False)

    # -------- lifecycle --------

    def _ensure_index(self, dim: int) -> None:
        faiss = _get_faiss()
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
            self.dim = dim
        else:
            if self.dim is None:
                self.dim = dim
            if self.dim != dim:
                raise ValueError(f"Vector dimension mismatch: store={self.dim}, got={dim}") from None

    @property
    def count(self) -> int:
        """Return the number of chunks in the store."""
        return len(self._ids)

    # -------- VectorStore API --------

    def upsert(self, chunks: Sequence[EmbeddedChunk]) -> None:
        if not chunks:
            return

        first_vec = chunks[0].embedding
        if not isinstance(first_vec, np.ndarray):
            raise TypeError("EmbeddedChunk.embedding must be a numpy.ndarray")
        if first_vec.ndim != 1:
            raise ValueError("EmbeddedChunk.embedding must be a 1D vector")
        self._ensure_index(int(first_vec.shape[0]))

        # Separate additions vs updates
        to_add_vecs: list[np.ndarray] = []
        to_add_ids: list[str] = []

        updated_ids_set: set[str] = set()
        updated_vecs: list[np.ndarray] = []
        updated_ids: list[str] = []

        for ec in chunks:
            cid = ec.chunk.id
            vec = ec.embedding.astype(np.float32, copy=False)
            if cid in self._chunks:
                updated_ids_set.add(cid)
                updated_vecs.append(vec)
                updated_ids.append(cid)
                self._chunks[cid] = ec
            else:
                to_add_ids.append(cid)
                to_add_vecs.append(vec)
                self._chunks[cid] = ec

        # Apply updates by rebuilding (flat index is cheap enough to rebuild for moderate sizes)
        if updated_ids_set:
            assert self._index is not None
            survivor_ids: list[str] = []
            survivor_vecs: list[np.ndarray] = []

            for cid in self._ids:
                if cid not in updated_ids_set:
                    survivor_ids.append(cid)
                    survivor_vecs.append(self._chunks[cid].embedding.astype(np.float32, copy=False))

            self._index.reset()
            self._ids = survivor_ids.copy()
            if survivor_vecs:
                V = np.stack(survivor_vecs, axis=0)
                self._index.add(V)

            if updated_vecs:
                Vupd = np.stack(updated_vecs, axis=0)
                self._index.add(Vupd)
                self._ids.extend(updated_ids)

        # Apply additions
        if to_add_vecs:
            assert self._index is not None
            Vnew = np.stack(to_add_vecs, axis=0)
            self._index.add(Vnew)
            self._ids.extend(to_add_ids)

    def search(self, query_vec: np.ndarray, top_k: int) -> SearchResult:
        if self._index is None or not self._ids:
            return SearchResult(query=None, hits=[])

        if query_vec.ndim != 1:
            raise ValueError("query_vec must be a 1D vector")
        q = query_vec.astype(np.float32, copy=False)

        Q = q.reshape(1, -1)
        assert self._index is not None
        scores, idxs = self._index.search(Q, k=min(top_k, len(self._ids)))
        ids = [self._ids[i] for i in idxs[0] if i != -1]

        hits: list[Hit] = []
        for i, cid in enumerate(ids):
            ec = self._chunks[cid]
            hits.append(Hit(
                doc_id=ec.chunk.doc_id,
                chunk_id=ec.chunk.id,
                chunk_order=ec.chunk.order,
                score=float(scores[0][i]),
                snippet=ec.chunk.text[:100] + "..." if len(ec.chunk.text) > 100 else ec.chunk.text,
                chunk=ec.chunk,
                metadata=ec.chunk.metadata
            ))

        return SearchResult(query=None, hits=hits)

    # -------- Persistence --------

    def save(self, dir_path: str | os.PathLike, *, model_name: str | None = None) -> None:
        """Persist index, embeddings, ids, and chunk metadata to a directory."""
        if self._index is None or not self._ids:
            # Empty store: create dir and write empty metadata
            d = Path(dir_path)
            d.mkdir(parents=True, exist_ok=True)
            meta = {
                "version": 1,
                "dim": self.dim or 0,
                "count": 0,
                "ids": [],
                "model_name": model_name,
            }
            _atomic_write_json(d / "meta.json", meta)
            # create empty files
            (d / "chunks.jsonl").write_text("", encoding="utf-8")
            np.save(d / "embeddings.npy", np.zeros((0, self.dim or 0), dtype=np.float32))
            # index.faiss may be omitted for empty; keep consistent:
            _atomic_write_faiss(d / "index.faiss", self._index)
            return

        # Gather embeddings in current order of _ids
        V = np.stack([self._chunks[cid].embedding.astype(np.float32, copy=False) for cid in self._ids], axis=0)
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)

        # Write embeddings, chunks metadata, faiss index, and meta.json atomically
        np.save(d / "embeddings.npy", V)
        _atomic_write_faiss(d / "index.faiss", self._index)

        # chunks.jsonl: minimal info for search results (embedding not stored here)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(d), delete=False) as tmp:
            tmp_path = Path(tmp.name)
            for cid in self._ids:
                ec = self._chunks[cid]
                rec = {"id": cid, "text": ec.chunk.text, "meta": ec.chunk.metadata}
                tmp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        (d / "chunks.jsonl").unlink(missing_ok=True)
        tmp_path.replace(d / "chunks.jsonl")

        meta = {
            "version": 1,
            "dim": int(self.dim or V.shape[1]),
            "count": int(len(self._ids)),
            "ids": self._ids,
            "model_name": model_name,
        }
        _atomic_write_json(d / "meta.json", meta)

    @classmethod
    def load(cls, dir_path: str | os.PathLike, *, expected_model_name: str | None = None) -> FaissVectorStore:
        """Load store from a directory. Validates dimensions and count."""
        faiss = _get_faiss()
        d = Path(dir_path)
        meta = _read_json(d / "meta.json")
        version = int(meta.get("version", 0))
        if version != 1:
            raise ValueError(f"Unsupported index version: {version}")

        dim = int(meta["dim"])
        ids: list[str] = list(meta["ids"])
        count = int(meta["count"])
        model_name = meta.get("model_name")

        if expected_model_name is not None and model_name is not None:
            if expected_model_name != model_name:
                raise ValueError(f"Model mismatch: expected={expected_model_name}, stored={model_name}")

        # Load embeddings (source of truth)
        emb_path = d / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"Missing embeddings file: {emb_path}")
        V = np.load(emb_path).astype(np.float32, copy=False)
        if V.ndim != 2 or V.shape[1] != dim:
            raise ValueError(f"Embeddings shape mismatch: got {V.shape}, expected (*, {dim})")
        if V.shape[0] != len(ids) or count != len(ids):
            raise ValueError(f"Count/ids mismatch: meta.count={count}, ids={len(ids)}, embeddings={V.shape[0]}")

        # Try load FAISS index; if fails, rebuild from embeddings
        idx_path = d / "index.faiss"
        index = None
        if idx_path.exists():
            try:
                index = faiss.read_index(str(idx_path))
                if getattr(index, "d", dim) != dim:
                    index = None  # fallback to rebuild
            except Exception:
                index = None

        if index is None:
            index = faiss.IndexFlatIP(dim)
            if V.shape[0] > 0:
                index.add(V)

        # Read chunks.jsonl to rebuild chunk metadata
        chunks_path = d / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"Missing chunks metadata: {chunks_path}")

        chunks_map: dict[str, EmbeddedChunk] = {}
        with chunks_path.open("r", encoding="utf-8") as f:
            for _i, line in enumerate(f):
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec["id"]
                text = rec.get("text", "")
                meta_rec = rec.get("meta", {})
                # build EmbeddedChunk w/ embedding from V at aligned position
                try:
                    pos = ids.index(cid)
                except ValueError:
                    raise ValueError(f"id {cid!r} not found in ids list") from None

                # local import to avoid cycles
                from ai_obsidian_service.core import Chunk, ChunkId, DocId

                emb = V[pos, :]
                # reconstruct Chunk from serialized minimal metadata
                order = int(meta_rec.get("order", 0)) if isinstance(meta_rec, dict) else 0
                source_id = meta_rec.get("sourceId", "unknown") if isinstance(meta_rec, dict) else "unknown"
                chunks_map[cid] = EmbeddedChunk(
                    chunk=Chunk(
                        id=ChunkId(cid),
                        doc_id=DocId(source_id),
                        order=order,
                        text=text,
                        metadata=meta_rec if isinstance(meta_rec, dict) else {},
                    ),
                    embedding=emb,
                )

        # Assemble store
        store = cls(dim=dim)
        store._index = index
        store._ids = ids
        store._chunks = chunks_map
        return store


# -------- helpers for atomic writes --------

def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    path.unlink(missing_ok=True)
    tmp_path.replace(path)


def _atomic_write_faiss(path: Path, index: Any) -> None:  # faiss.Index | None
    faiss = _get_faiss()
    path.parent.mkdir(parents=True, exist_ok=True)
    # For empty store, still write an empty index container for consistency
    idx = index or faiss.IndexFlatIP(0)
    with tempfile.NamedTemporaryFile("wb", dir=str(path.parent), delete=False) as tmp:
        tmp_path = Path(tmp.name)
        faiss.write_index(idx, str(tmp_path))
    path.unlink(missing_ok=True)
    tmp_path.replace(path)