import os
import tempfile
import time
from datetime import datetime as _dt
from pathlib import Path

import faiss
import numpy as np


def _ts(msg: str):
    print(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")


class FaissIndex:
    def __init__(self, dim: int, path: Path):
        self.dim = dim
        self.path = Path(path)
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vecs: np.ndarray):
        self.index.add(vecs)

    def save(self):
        _ts(f"[faiss] saving index → {self.path}")
        t0 = time.perf_counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self.path.parent), prefix="faiss.", suffix=".tmp"
        )
        os.close(fd)
        try:
            faiss.write_index(self.index, tmp_path)
            os.replace(tmp_path, str(self.path))
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        dur = time.perf_counter() - t0
        try:
            size = os.path.getsize(self.path)
        except Exception:
            size = -1
        _ts(f"[faiss] saved in {dur:.1f}s, size={size / 1e6:.2f} MB")

    @classmethod
    def load(cls, path: Path):
        index = faiss.read_index(str(path))
        fi = cls(index.d, path)
        fi.index = index
        return fi
