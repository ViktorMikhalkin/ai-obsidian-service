from typing import Tuple, List
from pathlib import Path
import yaml
from .models import Citation, SearchHit
from .llm_ollama import ollama_generate

_cfg = yaml.safe_load(Path("config.yaml").read_text())
_LLM_MODEL = _cfg.get("llm", {}).get("model", "qwen2.5:7b-instruct")
_MODE = _cfg.get("rag", {}).get("mode", "generative")

SYSTEM_PROMPT = (
    "You are an assistant that must answer strictly based on the provided excerpts.\n"
    "- Be concise\n- Do not hallucinate beyond the excerpts\n"
    "- At the end, list sources as [n] path#anchor.\n"
)


def _build_prompt(query: str, hits: List[SearchHit]) -> str:
    ctx_lines = []
    for i, h in enumerate(hits[:6], 1):
        ctx_lines.append(f"[{i}] {h.doc_path}\n{h.preview}\n")
    context = "\n".join(ctx_lines)
    return f"{SYSTEM_PROMPT}\nQuestion: {query}\n\nContext:\n{context}\n\nAnswer:"


def _extractive(hits: List[SearchHit]) -> Tuple[str, List[Citation]]:
    cits = [
        Citation(doc_path=h.doc_path, chunk_id=h.chunk_id, snippet=h.preview, span=None)
        for h in hits[:6]
    ]
    if not cits:
        return (
            "No relevant excerpts found. Rebuild the index or refine your query.",
            [],
        )
    ans = "Found excerpts (extractive):\n" + "\n".join(
        f"- {c.snippet}  [{c.doc_path}]" for c in cits
    )
    return ans, cits


def answer_with_citations(
    query: str, hits: List[SearchHit]
) -> Tuple[str, List[Citation]]:
    if not hits:
        return ("No relevant excerpts found.", [])
    if _MODE == "extractive":
        return _extractive(hits)
    prompt = _build_prompt(query, hits)
    try:
        text = ollama_generate(prompt, model=_LLM_MODEL)
    except Exception as e:
        fallback, cits = _extractive(hits)
        return f"(LLM unavailable: {e})\n\n{fallback}", cits
    cits = [
        Citation(doc_path=h.doc_path, chunk_id=h.chunk_id, snippet=h.preview, span=None)
        for h in hits[:6]
    ]
    return text, cits
