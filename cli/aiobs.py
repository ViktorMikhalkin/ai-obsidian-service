import sys
import os
import json
import yaml
import fnmatch
import typer
import typer as _typer
from pathlib import Path
from datetime import datetime as _dt

from indexer.parsers.md_parser import parse_markdown
from indexer.parsers.pdf_parser import extract_pdf_per_pages
from indexer.parsers.epub_parser import iter_epub_docs
from indexer.parsers.chunker import chunk_text
from indexer.embedder import Embedder
from indexer.store.vector_faiss import FaissIndex


def _ts(msg: str):
    """
    Plain, stable logging (no progress bars, no terminal control codes).
    Printed lines are friendly to TTY, conda run, CI logs, and IDE consoles.
    """
    _typer.echo(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")


app = typer.Typer(help="AI↔Obsidian CLI v4.3.0")


def load_config():
    """
    Load config.yaml and apply safe test-mode overrides if AIOBS_TEST_MODE=1.
    Also prints a short summary of key settings.
    """
    with open("config.yaml", "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f.read())

    # --- SAFETY MODE FOR TESTS / CI ---
    # If AIOBS_TEST_MODE=1 use a safe temp index dir to avoid overwriting real data.
    if str(os.environ.get("AIOBS_TEST_MODE", "0")) == "1":
        import tempfile
        safe_root = os.environ.get("AIOBS_TEST_INDEX_DIR") or tempfile.mkdtemp(prefix="aiobs-test-")
        safe_index_dir = str(Path(safe_root) / "index")
        Path(safe_index_dir).mkdir(parents=True, exist_ok=True)

        _cfg["index_dir"] = safe_index_dir
        emb = _cfg.setdefault("embeddings", {})
        faiss_cfg = emb.setdefault("faiss", {})
        faiss_cfg["index_path"] = str(Path(safe_index_dir) / "faiss.index")
        faiss_cfg["dim_path"] = str(Path(safe_index_dir) / "dim.txt")
        emb.setdefault("device", "cpu")
        _ts(f"[test-mode] index_dir → {safe_index_dir}")

    _ts(
        f"[config] index_dir={_cfg.get('index_dir', 'index')} "
        f"model={_cfg.get('embeddings',{}).get('model')} "
        f"device={_cfg.get('embeddings',{}).get('device','cpu')} "
        f"batch_size={_cfg.get('embeddings',{}).get('batch_size', 64)} "
        f"dtype={_cfg.get('embeddings',{}).get('dtype','fp32')} "
        f"search_on={_cfg.get('embeddings',{}).get('faiss',{}).get('search_on','cpu')}"
    )

    return _cfg


def iter_files(root: Path, patterns):
    """
    Yield files under 'root' that match any glob from 'patterns' (case-insensitive).
    """
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            sfull = str(full).lower()
            if any(fnmatch.fnmatch(sfull, p.lower()) for p in patterns):
                yield full


def is_excluded(p: Path, excludes):
    """
    Return True if a path matches any exclude globs (case-insensitive).
    """
    s = str(p).lower()
    return any(fnmatch.fnmatch(s, ex.lower()) for ex in excludes)


def _fmt_eta(seconds: float) -> str:
    """
    Format seconds as H:MM:SS or M:SS.
    """
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


class _Ticker:
    """
    Time-based logger: emits at most once per 'interval' seconds.
    Use to avoid flooding logs with per-item lines.
    """
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


@app.command()
def build():
    """
    Build the vector index:
    - scan files (MD/PDF/EPUB)
    - parse and chunk content
    - embed chunks
    - write FAISS index and JSONL metadata

    Progress behavior (explicit):
    - No progress bars.
    - Log compact progress lines every few seconds with processed/total, rate, and ETA.
    - Final summary lines remain for each phase.
    """
    cfg = load_config()
    vault = Path(cfg["vault_path"])
    libraries = [Path(p) for p in cfg.get("library_paths", [])]
    include_globs = cfg["include_globs"]
    exclude_globs = cfg.get("exclude_globs", [])
    target_tokens = cfg["chunk"]["target_tokens"]
    overlap_tokens = cfg["chunk"]["overlap_tokens"]

    # Scan sources
    notes, pdfs, epubs = [], [], []
    sources = [vault] + libraries
    for src in sources:
        if not src.exists():
            continue
        for f in iter_files(src, include_globs):
            if is_excluded(f, exclude_globs):
                continue
            low = f.suffix.lower()
            if low == ".md":
                notes.append(f)
            elif low == ".pdf":
                pdfs.append(f)
            elif low == ".epub":
                epubs.append(f)

    _ts(f"[scan] found: {len(notes)} MD, {len(pdfs)} PDF, {len(epubs)} EPUB")

    # Collect chunks + metadata
    texts, metas = [], []

    def add_doc(kind, path_str, raw_text):
        for chunk, (a, b) in chunk_text(raw_text, target_tokens, overlap_tokens):
            preview = chunk[:200]
            metas.append({"path": path_str, "kind": kind, "span": [a, b], "preview": preview})
            texts.append(chunk)

    # Phase: MD
    import time as _t
    tick = _Ticker(2.0)
    total = len(notes)
    if total:
        t0 = _t.perf_counter()
        for i, md in enumerate(notes, 1):
            parsed = parse_markdown(md)
            if parsed and parsed.text:
                add_doc("md", str(md), parsed.text)
            if tick.should_log() or i == total:
                elapsed = max(1e-6, _t.perf_counter() - t0)
                rate = i / elapsed
                pct = (i * 100) // total
                eta = _fmt_eta((total - i) / rate if rate > 0 else 0)
                _ts(f"[MD] {i}/{total} ({pct}%) | {rate:.1f} files/s | ETA {eta}")

    # Phase: PDF
    total = len(pdfs)
    if total:
        t0 = _t.perf_counter()
        tick = _Ticker(2.0)
        done = 0
        for p in pdfs:
            try:
                for page_no, txt in extract_pdf_per_pages(p):
                    add_doc("pdf", f"{p}#page={page_no}", txt)
            except Exception:
                # Swallow errors to keep the run resilient
                pass
            done += 1
            if tick.should_log() or done == total:
                elapsed = max(1e-6, _t.perf_counter() - t0)
                rate = done / elapsed
                pct = (done * 100) // total
                eta = _fmt_eta((total - done) / rate if rate > 0 else 0)
                _ts(f"[PDF] {done}/{total} ({pct}%) | {rate:.1f} files/s | ETA {eta}")

    # Phase: EPUB
    total = len(epubs)
    if total:
        t0 = _t.perf_counter()
        tick = _Ticker(2.0)
        done = 0
        for e in epubs:
            try:
                for frag_id, txt in iter_epub_docs(e):
                    add_doc("epub", f"{e}#{frag_id}", txt)
            except Exception:
                # Swallow errors to keep the run resilient
                pass
            done += 1
            if tick.should_log() or done == total:
                elapsed = max(1e-6, _t.perf_counter() - t0)
                rate = done / elapsed
                pct = (done * 100) // total
                eta = _fmt_eta((total - done) / rate if rate > 0 else 0)
                _ts(f"[EPUB] {done}/{total} ({pct}%) | {rate:.1f} files/s | ETA {eta}")

    # Embedding configuration
    emb = cfg.get("embeddings", {})
    embed_model = emb.get("model", "intfloat/multilingual-e5-small")
    device = emb.get("device", "cpu")
    dtype = emb.get("dtype", "fp32")
    batch_size = int(emb.get("batch_size", 64) or 64)

    _ts(f"[embed] init model: {embed_model} (device={device}, dtype={dtype})")
    embed = Embedder(embed_model, device=device, dtype=dtype)
    _ts("[embed] model ready")

    # Ensure index dir
    index_dir = Path(cfg.get("index_dir", "index"))
    index_dir.mkdir(parents=True, exist_ok=True)

    if not texts:
        (index_dir / "dim.txt").write_text("0", encoding="utf-8")
        typer.echo("[build] No text to index")
        return

    # Phase: EMBED (chunks)
    _ts("[embed] encoding chunks...")
    t0 = _t.perf_counter()
    N = len(texts)
    acc = []
    tick = _Ticker(2.0)
    processed = 0

    for i in range(0, N, batch_size):
        batch = texts[i:i + batch_size]
        vec = embed.encode(batch, batch_size=batch_size)
        acc.append(vec)
        processed += len(batch)
        if tick.should_log() or processed == N:
            elapsed = max(1e-6, _t.perf_counter() - t0)
            rate = processed / elapsed
            pct = (processed * 100) // N
            eta = _fmt_eta((N - processed) / rate if rate > 0 else 0)
            _ts(f"[EMBED] {processed}/{N} ({pct}%) | {rate:.1f} chunks/s | ETA {eta}")

    # Finalize vectors
    import numpy as _np
    vecs = _np.vstack(acc)
    dur = _t.perf_counter() - t0
    _ts(f"[embed] done: {len(texts)} vectors in {dur:.1f}s, dim={vecs.shape[1]}")

    # Build and save FAISS index
    fa = FaissIndex(dim=vecs.shape[1], path=index_dir / "faiss.index")
    fa.add(vecs)
    fa.save()

    # Write metadata and dim
    with (index_dir / "index.jsonl").open("w", encoding="utf-8") as out:
        for i, m in enumerate(metas):
            m["id"] = i
            out.write(json.dumps(m, ensure_ascii=False) + "\n")
    (index_dir / "dim.txt").write_text(str(vecs.shape[1]), encoding="utf-8")
    _ts(f"[write] metadata → {index_dir / 'index.jsonl'}")
    _ts(f"[write] dim → {index_dir / 'dim.txt'}")
    typer.echo(f"[build] chunks: {len(texts)}  dim: {vecs.shape[1]}")

@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """
    Run the API with autoreload for local development.
    """
    import uvicorn
    uvicorn.run("indexer.app:app", host=host, port=port, reload=True)

@app.command()
def status():
    """
    Print number of chunks recorded in the on-disk JSONL metadata.
    """
    idx = Path("index/index.jsonl")
    n = sum(1 for _ in idx.open()) if idx.exists() else 0
    typer.echo(f"[status] chunks: {n}")

if __name__ == "__main__":
    app()