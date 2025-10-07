from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest

# Safe defaults for unit runs
os.environ.setdefault("AIOBS_TEST_MODE", "1")
os.environ.setdefault("VECTOR_STORE_BACKEND", "memory")
os.environ.setdefault("OLLAMA_BASE_URL", "")
os.environ.setdefault("OLLAMA_MODEL", "")


# --- Capability probes & skips for FAISS CPU (integration tests) --------------
def _has_faiss_cpu() -> bool:
    """Check if FAISS CPU is available."""
    # Skip FAISS check in test mode to avoid segfaults during collection
    if os.environ.get("AIOBS_TEST_MODE") == "1":
        return False
    try:
        return True
    except Exception:
        return False


HAS_FAISS_CPU = _has_faiss_cpu()


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    skip_faiss = pytest.mark.skip(reason="FAISS (CPU) not available")
    for item in items:
        if (
            ("integration_cpu" in item.keywords)
            or ("faiss" in item.keywords)
            or ("requires_faiss" in item.keywords)
        ):
            if not HAS_FAISS_CPU:
                item.add_marker(skip_faiss)


# --- Helpers ------------------------------------------------------------------
def _iter_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    import fnmatch
    import os as _os

    for dirpath, _, filenames in _os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            s = str(p).lower()
            if any(fnmatch.fnmatch(s, pat.lower()) for pat in patterns):
                yield p


# --- Fixtures -----------------------------------------------------------------
@pytest.fixture
def mini_vault(tmp_path: Path) -> Path:
    """
    Minimal Obsidian-like vault:
      vault/
        note1.md
        sub/note2.md
    """
    root = tmp_path / "vault"
    (root / "sub").mkdir(parents=True, exist_ok=True)
    (root / "note1.md").write_text("# Note 1\nHello world\n", encoding="utf-8")
    (root / "sub" / "note2.md").write_text("# Note 2\nHello again\n", encoding="utf-8")
    return root


@pytest.fixture
def search_service(mini_vault: Path):
    """
    Build SearchService (memory backend) and index the mini_vault (MD only for unit).
    """
    from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
    from ai_obsidian_service.adapters.parsers import all_parsers
    from ai_obsidian_service.di_selector import make_components

    chunker = SimpleChunker(max_chars=1000, overlap=100)
    di = make_components(chunker=chunker)
    service = di.search  # unified SearchService

    try:
        service.parsers = all_parsers()
    except Exception:
        pass

    for p in _iter_files(mini_vault, patterns=["**/*.md"]):
        service.index_path(str(p))

    return service


@pytest.fixture
def empty_search_service():
    """
    Build SearchService (memory backend), but DO NOT index anything.
    Useful for tests that need a truly empty index.
    """
    from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
    from ai_obsidian_service.adapters.parsers import all_parsers
    from ai_obsidian_service.di_selector import make_components

    chunker = SimpleChunker(max_chars=1000, overlap=100)
    di = make_components(chunker=chunker)
    service = di.search

    try:
        service.parsers = all_parsers()
    except Exception:
        pass

    return service
