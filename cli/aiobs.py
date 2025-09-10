import typer, fnmatch, os, json, yaml
from pathlib import Path
from indexer.parsers.md_parser import parse_markdown
from indexer.parsers.pdf_parser import extract_pdf_per_pages
from indexer.parsers.epub_parser import iter_epub_docs
from indexer.parsers.chunker import chunk_text
from indexer.embedder import Embedder
from indexer.store.vector_faiss import FaissIndex
from tqdm import tqdm
from datetime import datetime as _dt
import typer as _typer

def _ts(msg: str):
    _typer.echo(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")

app = typer.Typer(help="AI↔Obsidian CLI v4.3.0")

def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f.read())

    # --- SAFETY MODE FOR TESTS / CI ---
    # If AIOBS_TEST_MODE=1 change index_dir and paths FAISS to the temp safe ones
    # to avoid breaking real indexes while testing.

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
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            full = Path(dirpath) / name
            sfull = str(full).lower()
            if any(fnmatch.fnmatch(sfull, p.lower()) for p in patterns):
                yield full

def is_excluded(p: Path, excludes):
    s = str(p).lower()
    return any(fnmatch.fnmatch(s, ex.lower()) for ex in excludes)

@app.command()
def build():
    cfg = load_config()
    vault = Path(cfg["vault_path"])
    libraries = [Path(p) for p in cfg.get("library_paths", [])]
    include_globs = cfg["include_globs"]
    exclude_globs = cfg.get("exclude_globs", [])
    target_tokens = cfg["chunk"]["target_tokens"]
    overlap_tokens = cfg["chunk"]["overlap_tokens"]

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

    texts, metas = [], []
    def add_doc(kind, path_str, raw_text):
        for chunk, (a, b) in chunk_text(raw_text, target_tokens, overlap_tokens):
            preview = chunk[:200]
            metas.append({"path": path_str, "kind": kind, "span": [a, b], "preview": preview})
            texts.append(chunk)

    for md in tqdm(notes, desc="MD"):
        parsed = parse_markdown(md)
        if parsed and parsed.text:
            add_doc("md", str(md), parsed.text)

    for p in tqdm(pdfs, desc="PDF"):
        try:
            for page_no, txt in extract_pdf_per_pages(p):
                add_doc("pdf", f"{p}#page={page_no}", txt)
        except Exception:
            pass

    for e in tqdm(epubs, desc="EPUB"):
        try:
            for frag_id, txt in iter_epub_docs(e):
                add_doc("epub", f"{e}#{frag_id}", txt)
        except Exception:
            pass

    embed_model = cfg.get("embeddings", {}).get("model", "intfloat/multilingual-e5-small")
    device = cfg.get("embeddings", {}).get("device", "cpu")
    dtype = cfg.get("embeddings", {}).get("dtype", "fp32")
    batch_size = int(cfg.get("embeddings", {}).get("batch_size", 64) or 64)

    _ts(f"[embed] init model: {embed_model} (device={device}, dtype={dtype})")
    embed = Embedder(embed_model, device=device, dtype=dtype)
    _ts("[embed] model ready")

    index_dir = Path(cfg.get("index_dir", "index"))
    index_dir.mkdir(parents=True, exist_ok=True)

    if not texts:
        (index_dir / "dim.txt").write_text("0", encoding="utf-8")
        typer.echo("[build] No text to index")
        return

    _ts("[embed] encoding chunks...")
    import time as _t
    t0 = _t.perf_counter()
    N = len(texts)
    acc = []
    for i in range(0, N, batch_size):
        batch = texts[i:i + batch_size]
        vec = embed.encode(batch, batch_size=batch_size)
        acc.append(vec)
        if ((i // batch_size) % 10) == 0 or i + batch_size >= N:
            _ts(f"[embed] progress: {i + len(batch)}/{N} ({((i + len(batch)) * 100) // max(1, N)}%)")

    import numpy as _np
    vecs = _np.vstack(acc)
    dur = _t.perf_counter() - t0
    _ts(f"[embed] done: {len(texts)} vectors in {dur:.1f}s, dim={vecs.shape[1]}")

    fa = FaissIndex(dim=vecs.shape[1], path=index_dir / "faiss.index")
    fa.add(vecs)
    fa.save()

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
    import uvicorn
    uvicorn.run("indexer.app:app", host=host, port=port, reload=True)

# ... existing code ...
@app.command()
def status():
    idx = Path("index/index.jsonl")
    n = sum(1 for _ in idx.open()) if idx.exists() else 0
    typer.echo(f"[status] chunks: {n}")

if __name__ == "__main__":
    app()
