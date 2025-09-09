def chunk_text(text: str, target_tokens: int = 800, overlap_tokens: int = 120):
    toks = text.split()
    if len(toks) <= target_tokens:
        yield text, (0, len(toks))
        return
    step = max(1, target_tokens - overlap_tokens)
    i = 0
    while i < len(toks):
        j = min(i + target_tokens, len(toks))
        yield " ".join(toks[i:j]), (i, j)
        i += step
