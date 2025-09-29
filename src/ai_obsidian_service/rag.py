from __future__ import annotations

from ai_obsidian_service.adapters.llm.ollama_client import OllamaClient, OllamaError
from ai_obsidian_service.domain.models import SearchResult


def _find_span(haystack: str | None, needle: str) -> tuple[int, int]:
    if not haystack:
        return (-1, -1)
    i = haystack.find(needle or "")
    return (i, i + len(needle)) if i >= 0 and needle else (-1, -1)

def _compose_context(result: SearchResult, *, max_snippets: int = 8) -> str:
    lines: list[str] = []
    for h in result.hits[:max_snippets]:
        path = None
        try:
            if hasattr(h, "chunk") and h.chunk is not None and hasattr(h.chunk, "meta"):
                meta = h.chunk.meta
                if meta is not None:
                    path = meta.get("path")
        except Exception:
            path = None
        label = f"[{path or h.doc_id}:{h.chunk_id}]"
        snip = (h.snippet or "").strip().replace("\n", " ")
        if snip:
            lines.append(f"{label} {snip}")
    return "\n".join(lines)

def _default_prompt(query: str, context: str) -> str:
    return (
        "You are a helpful assistant. Answer strictly based on the provided context.\n"
        "If the answer is not present, say you don't know.\n\n"
        f"Question:\n{query}\n\nContext:\n{context}\n\nAnswer:\n"
    )

def _mini_answer_from_snippets(result: SearchResult) -> str:
    hits = result.hits[:]
    if not hits:
        return "I couldn't find relevant context. Try reindexing your vault or widening the query."
    bullets: list[str] = []
    seen: set[str] = set()
    for h in hits:
        snip = (h.snippet or "").strip()
        if not snip:
            continue
        key = snip.lower()
        if key in seen:
            continue
        seen.add(key)
        bullets.append(f"- {snip}")
        if len(bullets) >= 5:
            break
    header = f"Based on {len(hits)} retrieved chunks"
    if result.query and result.query.top_k:
        header += f" (top_k={result.query.top_k})"
    return header + ":\n" + "\n".join(bullets) if bullets else header + "."

def answer_with_citations(
        query: str,
        result: SearchResult,
        *,
        llm: OllamaClient | None = None,
        system_prompt: str | None = None,
) -> tuple[str, list[dict]]:
    # citations
    citations: list[dict] = []
    for h in result.hits[:10]:
        chunk_text = getattr(h, "chunk_text", None)
        if not chunk_text and hasattr(h, "chunk") and h.chunk is not None:
            try:
                chunk_text = h.chunk.text
            except Exception:
                chunk_text = None
        span = _find_span(chunk_text, h.snippet or "")
        doc_path = None
        try:
            if hasattr(h, "chunk") and h.chunk is not None and hasattr(h.chunk, "meta"):
                meta = h.chunk.meta
                if meta is not None:
                    doc_path = meta.get("path")
        except Exception:
            doc_path = None
        citations.append(
            {"doc_path": doc_path, "chunk_id": str(h.chunk_id), "snippet": h.snippet or "", "span": span}
        )

    if llm is None:
        return (_mini_answer_from_snippets(result), citations)

    top_k = result.query.top_k if result.query else 8
    ctx = _compose_context(result, max_snippets=max(1, top_k or 8))
    if not ctx.strip():
        return ("I couldn't find relevant context. Try reindexing your vault or widening the query.", citations)
    prompt = _default_prompt(query, ctx)
    try:
        text = llm.generate(prompt, system=system_prompt)
    except OllamaError:
        text = _mini_answer_from_snippets(result)
    return (text.strip(), citations)
