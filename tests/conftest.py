from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType
from typing import Any

import pytest


# -------------------- helpers: feature detection --------------------

def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False

def _has_faiss() -> bool:
    # Одинаково для faiss-cpu и faiss-gpu: модуль называется "faiss"
    return _has_module("faiss")

def _has_st() -> bool:
    return _has_module("sentence_transformers")

def _has_gpu() -> bool:
    # Простая эвристика: есть torch + cuda
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# -------------------- optional global fakes (default ON) --------------------

USE_FAKE_FAISS = os.getenv("USE_FAKE_FAISS", "1") == "1"
USE_FAKE_ST = os.getenv("USE_FAKE_ST", "1") == "1"

def _install_fake_faiss() -> None:
    """
    Мини-мок FAISS для быстрых тестов:
      - IndexFlatIP(d) с .add/.reset/.search
      - write_index/read_index (заглушки)
    """
    if "faiss" in sys.modules:
        return
    class _IndexFlatIP:
        def __init__(self, d: int):
            self.d = int(d)
            self._vecs = None  # type: Any

        def add(self, X):
            import numpy as np
            X = np.asarray(X, dtype="float32")
            if self._vecs is None:
                self._vecs = X
            else:
                self._vecs = np.vstack([self._vecs, X])

        def reset(self):
            self._vecs = None

        def search(self, Q, k: int):
            import numpy as np
            Q = np.asarray(Q, dtype="float32")
            if self._vecs is None or (hasattr(self._vecs, "size") and self._vecs.size == 0):
                return (np.zeros((len(Q), k), dtype="float32"),
                        -1 * np.ones((len(Q), k), dtype="int64"))
            V = self._vecs
            # косинус через скалярное (ожидаем нормализованные векторы)
            scores = Q @ V.T
            idxs = np.argsort(-scores, axis=1)[:, :k]
            take_scores = np.take_along_axis(scores, idxs, axis=1)
            return take_scores.astype("float32"), idxs.astype("int64")

    m = ModuleType("faiss")
    m.IndexFlatIP = _IndexFlatIP  # type: ignore[attr-defined]

    def write_index(index, path: str):
        with open(path, "wb") as f:
            f.write(b"FAISS_MOCK")

    def read_index(path: str):
        return _IndexFlatIP(0)

    m.write_index = write_index  # type: ignore[attr-defined]
    m.read_index = read_index    # type: ignore[attr-defined]
    sys.modules["faiss"] = m

def _install_fake_st() -> None:
    if "sentence_transformers" in sys.modules:
        return
    class _Model:
        def __init__(self, name: str):
            self.name = name
        def encode(self, text: Any, convert_to_numpy: bool = True, normalize_embeddings: bool = True):
            import numpy as np
            def enc_one(t: str) -> Any:
                arr = np.array([(abs(hash(t)) >> (i*8)) & 0xFF for i in range(4)], dtype="float32")
                if normalize_embeddings:
                    n = np.linalg.norm(arr) + 1e-12
                    arr = arr / n
                return arr
            if isinstance(text, str):
                return enc_one(text)
            return [enc_one(t) for t in text]
    pkg = ModuleType("sentence_transformers")
    pkg.SentenceTransformer = _Model  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = pkg


# Ставим фейки по умолчанию (для unit/e2e(memory)).
# Для интеграционных прогонов выставляй USE_FAKE_FAISS=0, USE_FAKE_ST=0.
if USE_FAKE_FAISS:
    _install_fake_faiss()
if USE_FAKE_ST:
    _install_fake_st()


# -------------------- auto-skip by markers --------------------

def pytest_collection_modifyitems(config, items):
    # Если реальные зависимости нужны, но недоступны — пропускаем помеченные тесты.
    has_faiss = _has_faiss()
    has_st = _has_st()
    has_gpu = _has_gpu()

    skip_faiss = pytest.mark.skip(reason="requires FAISS (set USE_FAKE_FAISS=0 and install faiss-cpu/faiss-gpu)")
    skip_st = pytest.mark.skip(reason="requires sentence-transformers (set USE_FAKE_ST=0 and install package)")
    skip_gpu = pytest.mark.skip(reason="requires GPU (torch.cuda.is_available() == True)")

    for item in items:
        # интеграционные тесты на CPU просят реальные FAISS/ST
        if "integration_cpu" in item.keywords:
            if not has_faiss:
                item.add_marker(skip_faiss)
            if not has_st:
                item.add_marker(skip_st)
        # интеграционные тесты на GPU
        if "integration_gpu" in item.keywords:
            if not has_faiss:
                item.add_marker(skip_faiss)
            if not has_st:
                item.add_marker(skip_st)
            if not has_gpu:
                item.add_marker(skip_gpu)
        # точечные зависимости
        if "requires_faiss" in item.keywords and not has_faiss:
            item.add_marker(skip_faiss)
        if "requires_st" in item.keywords and not has_st:
            item.add_marker(skip_st)
