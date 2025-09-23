from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from datetime import datetime as _dt
from pathlib import Path

import typer
import yaml

from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.index.faiss_index import FaissIndex
from ai_obsidian_service.adapters.parsers import default_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService

app = typer.Typer(help="AI↔Obsidian CLI (unified ports)")

# ---------- small logging helpers ----------

def _ts(msg: str) -> None:
    """Plain, stable logging (friendly to TTY/CI)."""
    typer.echo(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")


class _Ticker:
    """Emit logs at most once per 'interval' seconds."""
    def __init__(self, interval_sec: float = 2.0):
        import time as _t
        self.interval = interval_sec
        self._last = _t.perf_counter()

    def should_log(self) -> bool:
        import time as _t
        now = _t.perf_counter()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"


# ---------- config & scan ----------

def load_config() -> dict:
    """
    Load config.yaml and apply safe test-mode overrides if AIOBS_TEST_MODE=1.
    """
    with open("config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f.read()) or {}

    # SAFETY for CI/tests
    if str(os.environ.get("AIOBS_TEST_MODE", "0")) == "1":
        import tempfile
        safe_root = os.environ.get("AIOBS_TEST_INDEX_DIR") or tempfile.mkdtemp(prefix="aiobs-test-")
        safe_index_dir = str(Path(safe_root) / "index")
        Path(safe_index_dir).mkdir(parents=True, exist_ok=True)
        cfg["index_dir"] = safe_index_dir
        _ts(f"[test-mode] index_dir → {safe_index_dir}")

    emb = cfg.get("embeddings", {})
    _ts(
        f"[config] index_dir={cfg.get('index_dir', 'index')} "
        f"model={emb.get('model')} device={emb.get('device', 'cpu')} "
        f"batch_size={emb.get('batch_size', 64)} dtype={emb.get('dtype', 'fp32')} "
        f"search_on={emb.get('faiss', {}).get('search_on', 'cpu')}"
    )
    return cfg


def iter_files(root: Path, patterns: Iterable[str]) -> Iterable[Path]:
    """Yield files under 'root' matching any glob from 'patterns' (case-insensitive)."""
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            sfull = str(full).lower()
            if any(fnmatch.fnmatch(sfull, p.lower()) for p in patterns):
                yield full


def is_excluded(p: Path, excludes: Iterable[str]) -> bool:
    s = str(p).lower()
    return any(fnmatch.fnmatch(s, ex.lower()) for ex in excludes)


# ---------- DI builder (inline) ----------

def _build_search_service(index_dir: str | None, *, max_chars: int, overlap: int) -> SearchService:
    """
    Build SearchService with unified ports:
      - parsers: default_parsers()
      - chunker: SimpleChunker(max_chars, overlap)
      - index  : FaissIndex(index_dir, dim=64)
    """
    parsers = default_parsers()
    chunker = SimpleChunker(max_chars=max_chars, overlap=overlap)
    index = FaissIndex(index_dir=index_dir, dim=64)
    return SearchService(parsers=parsers, chunker=chunker, index=index)


# ---------- commands ----------

@app.command()
def build() -> None:
    """
    Bulk-index:
    - scan MD/PDF/EPUB files by config globs,
    - delegate parsing+chunking+upsert в SearchService.
    """
    import time as _t

    cfg = load_config()
    vault = Path(cfg["vault_path"])
    libraries = [Path(p) for p in cfg.get("library_paths", [])]
    include_globs = cfg.get("include_globs", ["**/*.md", "**/*.pdf", "**/*.epub"])
    exclude_globs = cfg.get("exclude_globs", [])

    # map token settings to chars roughly (4 chars ≈ 1 токен), чтобы не ломать конфиг
    chunk_cfg = cfg.get("chunk", {})
    target_tokens = int(chunk_cfg.get("target_tokens", 250))
    overlap_tokens = int(chunk_cfg.get("overlap_tokens", 50))
    max_chars = max(50, target_tokens * 4)
    overlap = max(0, overlap_tokens * 4)

    index_dir = cfg.get("index_dir", "index")
    service = _build_search_service(index_dir=index_dir, max_chars=max_chars, overlap=overlap)

    # collect files
    files: list[Path] = []
    for src in [vault, *libraries]:
        if not src.exists():
            continue
        for f in iter_files(src, include_globs):
            if not is_excluded(f, exclude_globs):
                files.append(f)

    if not files:
        _ts("[scan] nothing to index")
        return

    # stats/progress
    files.sort()
    total = len(files)
    _ts(f"[scan] found {total} files")
    tick = _Ticker(2.0)
    t0 = _t.perf_counter()
    done = 0
    indexed_chunks = 0

    for p in files:
        try:
            indexed_chunks += service.index_path(str(p))
        except Exception:
            # keep run resilient; you may log exception here if needed
            pass
        done += 1
        if tick.should_log() or done == total:
            elapsed = max(1e-6, _t.perf_counter() - t0)
            rate = done / elapsed
            pct = (done * 100) // total
            eta = _fmt_eta((total - done) / rate if rate > 0 else 0)
            _ts(f"[index] {done}/{total} ({pct}%) | {rate:.1f} files/s | ETA {eta}")

    _ts(f"[done] files={done}  chunks={indexed_chunks}  index_dir={index_dir}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Run the API with autoreload for local development.
    """
    import uvicorn
    uvicorn.run("ai_obsidian_service.api.app:app", host=host, port=port, reload=True)


@app.command()
def status() -> None:
    """
    Print number of chunks recorded in the on-disk JSONL metadata (best-effort).
    """
    idx = Path("index/index.jsonl")
    n = sum(1 for _ in idx.open()) if idx.exists() else 0
    typer.echo(f"[status] chunks: {n}")


if __name__ == "__main__":
    app()
