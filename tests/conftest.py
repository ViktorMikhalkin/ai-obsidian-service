import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

# 1) Ensure repository root is on PYTHONPATH so `from indexer ...` works
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 2) Set safe test mode ENV before any app modules import config
_WORK_ROOT = Path(tempfile.mkdtemp(prefix="aiobs-test-"))
_INDEX_DIR = _WORK_ROOT / "index"
_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Minimal index files so API endpoints can operate without real data
(_INDEX_DIR / "index.jsonl").write_text("", encoding="utf-8")
(_INDEX_DIR / "dim.txt").write_text("8", encoding="utf-8")
(_INDEX_DIR / "faiss.index").write_bytes(b"\x00")

# Create minimal config.yaml in the temporary work dir
# YAML is a superset of JSON, safe to dump as JSON here
(_WORK_ROOT / "config.yaml").write_text(
    (
        "{"
        f'"index_dir": "{_INDEX_DIR.as_posix()}",'
        '"vault_path": "",'
        '"library_paths": [],'
        '"include_globs": ["**/*.md","**/*.pdf","**/*.epub"],'
        '"chunk": {"target_tokens": 16, "overlap_tokens": 4},'
        '"server": {"host": "127.0.0.1", "port": 8000},'
        '"embeddings": {'
        '"model": "intfloat/multilingual-e5-small",'
        '"device": "cpu",'
        '"batch_size": 8,'
        '"faiss": {'
        f'"index_path": "{(_INDEX_DIR / "faiss.index").as_posix()}",'
        f'"dim_path": "{(_INDEX_DIR / "dim.txt").as_posix()}"'
        "}"
        "}"
        "}"
    ),
    encoding="utf-8",
)

os.environ.setdefault("AIOBS_TEST_MODE", "1")
os.environ.setdefault("AIOBS_TEST_INDEX_DIR", str(_INDEX_DIR))

# 3) Mock heavy dependencies at import time, so real packages are never loaded
# Mock sentence_transformers
st_pkg = types.ModuleType("sentence_transformers")


class _DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(
        self, texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False
    ):
        import numpy as _np

        return _np.zeros((len(texts), 8), dtype=_np.float32)


st_pkg.SentenceTransformer = _DummySentenceTransformer  # type: ignore[attr-defined]
sys.modules["sentence_transformers"] = st_pkg

# Mock faiss
faiss_mod = types.ModuleType("faiss")


class _DummyIndexFlatIP:
    def __init__(self, d):
        self.d = d

    def add(self, arr):
        pass


def _dummy_write_index(index, path):
    Path(path).write_bytes(b"\x00")


def _dummy_read_index(path):
    return _DummyIndexFlatIP(8)


faiss_mod.IndexFlatIP = _DummyIndexFlatIP  # type: ignore[attr-defined]
faiss_mod.write_index = _dummy_write_index  # type: ignore[attr-defined]
faiss_mod.read_index = _dummy_read_index  # type: ignore[attr-defined]
sys.modules["faiss"] = faiss_mod


# 4) Switch CWD to the temp work dir so all relative paths resolve safely
@pytest.fixture(scope="session", autouse=True)
def _switch_cwd_to_work_root():
    old_cwd = os.getcwd()
    os.chdir(_WORK_ROOT)
    try:
        yield
    finally:
        os.chdir(old_cwd)
