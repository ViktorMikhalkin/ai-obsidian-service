from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import yaml
import traceback
import os
from contextlib import asynccontextmanager

# Optional Ollama client
try:
    import ollama  # type: ignore
except Exception:
    ollama = None

# Local imports
from indexer.embedder import Embedder
from indexer.store.vector_faiss import FaissIndex


# ---------------------------
# Pydantic models (API)
# ---------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchHit(BaseModel):
    id: int
    path: str
    kind: str
    preview: str
    score: float


class SearchResponse(BaseModel):
    results: list[SearchHit]


class AnswerRequest(BaseModel):
    query: str
    top_k: int = 5
    mode: str = "auto"  # "auto" (use ollama if available) or "extractive"
    model: str | None = None  # e.g., "llama3.1:8b"
    max_tokens: int = 256
    temperature: float = 0.2


class AnswerResponse(BaseModel):
    query: str
    answer: str
    sources: list[SearchHit]


# ---------------------------
# App state
# ---------------------------
CFG = {}
INDEX_DIR = None
FA = None
METAS: list[dict] = []
EMBED = None
DIM = None


# ---------------------------
# Helpers
# ---------------------------
def _load_config():
    """Load config.yaml from current working directory."""
    global CFG, INDEX_DIR
    with open("config.yaml", "r", encoding="utf-8") as f:
        CFG = yaml.safe_load(f)
    INDEX_DIR = Path(CFG.get("index_dir", "index"))
    return CFG


def _load_index():
    """Load FAISS index + metadata and dimension."""
    global FA, METAS, DIM
    idx_path = INDEX_DIR / "faiss.index"
    meta_path = INDEX_DIR / "index.jsonl"
    dim_path = INDEX_DIR / "dim.txt"

    if not idx_path.exists() or not meta_path.exists() or not dim_path.exists():
        raise FileNotFoundError(
            f"Missing index artifacts in {INDEX_DIR}. "
            f"Expected faiss.index, index.jsonl, dim.txt"
        )

    FA = FaissIndex.load(idx_path)
    DIM = int(Path(dim_path).read_text().strip())

    METAS = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                METAS.append(json.loads(line))
            except Exception:
                continue


def _init_embedder():
    """Initialize the embedding model wrapper."""
    global EMBED
    emb = CFG.get("embeddings", {}) or {}
    model = emb.get("model", "intfloat/multilingual-e5-small")
    device = emb.get("device", "cpu")
    dtype = emb.get("dtype", "fp32")
    EMBED = Embedder(model_name=model, device=device, dtype=dtype)


# ---------------------------
# Lifespan (startup/shutdown)
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: attempt to load config, index, and embedder.
    try:
        _load_config()
        _load_index()
        _init_embedder()
    except Exception:
        # Keep app running; /health will report problems
        traceback.print_exc()
    yield
    # Shutdown: nothing to clean up right now


# Create app with lifespan
app = FastAPI(title="AI↔Obsidian Search API", version="1.1.0", lifespan=lifespan)


# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
def health():
    errors = []
    if not CFG:
        errors.append("config_not_loaded")
    if FA is None:
        errors.append("faiss_not_loaded")
    if EMBED is None:
        errors.append("embedder_not_initialized")
    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "index_dir": str(INDEX_DIR) if INDEX_DIR else None,
        "chunks": len(METAS),
        "dim": DIM,
        "model": (CFG.get("embeddings", {}) or {}).get("model") if CFG else None,
        "device": (CFG.get("embeddings", {}) or {}).get("device") if CFG else None,
        "ollama_available": bool(ollama),
    }


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if FA is None or EMBED is None:
        raise HTTPException(status_code=503, detail="Service not ready. Check /health.")
    q = req.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    # E5 models benefit from "query: " prefix
    model_name = (CFG.get("embeddings", {}) or {}).get("model", "")
    q_to_encode = ("query: " + q) if "e5" in model_name.lower() else q

    try:
        qvec = EMBED.encode([q_to_encode], batch_size=1)
        D, indices = FA.index.search(qvec.astype("float32"), req.top_k)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Search error: {e}")

    results = []
    ids = indices[0].tolist()
    scores = D[0].tolist()
    for rid, score in zip(ids, scores):
        if 0 <= rid < len(METAS):
            m = METAS[rid]
            results.append(
                SearchHit(
                    id=rid,
                    path=m.get("path"),
                    kind=m.get("kind"),
                    preview=m.get("preview", "")[:300],
                    score=float(score),
                )
            )
    return SearchResponse(results=results)


def _extractive_summarize(query: str, hits: list[SearchHit]) -> str:
    """Fallback extractive answer if no LLM or no context."""
    if not hits:
        return "No relevant results were found."
    snippets = []
    for h in hits[:5]:
        p = h.preview.replace("\n", " ").strip()
        if len(p) > 400:
            p = p[:400] + "…"
        snippets.append(f"- [{h.path}] {p}")
    return (
        f"Query: {query}\n\n"
        f"Key points from top {len(hits)} results:\n"
        + "\n".join(snippets)
        + "\n\n(Generated without an LLM; this is an extractive summary of top passages.)"
    )


def _llm_summarize(
    query: str,
    hits: list[SearchHit],
    model: str | None,
    max_tokens: int,
    temperature: float,
) -> str:
    """Generate answer via Ollama if available; otherwise fallback to extractive summary."""
    if not ollama:
        return _extractive_summarize(query, hits)
    if not hits:
        return "No relevant results were found."

    ctx_parts = []
    for i, h in enumerate(hits[:8], 1):
        ctx_parts.append(f"[{i}] PATH: {h.path}\nPREVIEW: {h.preview}")
    context = "\n\n".join(ctx_parts)

    chosen_model = model or os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    prompt = (
        "You are a helpful assistant. Answer the user concisely using ONLY the context snippets "
        "below. Prefer concrete steps and short bullet points. If the answer is not in context, say so.\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT SNIPPETS:\n{context}\n\n"
        "RESPONSE (English, concise, include a short 'Why this matters' if appropriate):"
    )
    try:
        resp = ollama.chat(
            model=chosen_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return resp.get("message", {}).get(
            "content", ""
        ).strip() or _extractive_summarize(query, hits)
    except Exception:
        traceback.print_exc()
        return _extractive_summarize(query, hits)


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    sres = search(SearchRequest(query=req.query, top_k=req.top_k))
    hits = sres.results
    if req.mode == "extractive":
        ans = _extractive_summarize(req.query, hits)
    else:
        ans = _llm_summarize(
            req.query, hits, req.model, req.max_tokens, req.temperature
        )
    return AnswerResponse(query=req.query, answer=ans, sources=hits)
