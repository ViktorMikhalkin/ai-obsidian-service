from __future__ import annotations
from typing import Any, Optional
import httpx

class OllamaError(RuntimeError):
    pass

class OllamaClient:
    """
    Minimal sync client for Ollama /api/generate (non-stream).
    Env:
      - OLLAMA_BASE_URL=http://127.0.0.1:11434
      - OLLAMA_MODEL=llama3.1
      - OLLAMA_TIMEOUT=30
    """
    def __init__(self, base_url: str, model: str, *, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._client = httpx.Client(base_url=self.base_url, timeout=self.timeout_s)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def generate(self, prompt: str, *, system: Optional[str] = None) -> str:
        payload: dict[str, Any] = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            r = self._client.post("/api/generate", json=payload)
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama request failed: {e!r}") from e
        if r.status_code >= 400:
            raise OllamaError(f"Ollama error {r.status_code}: {r.text}")
        try:
            data = r.json()
        except Exception as e:
            raise OllamaError(f"Ollama bad JSON: {e!r}") from e
        text = str(data.get("response", "")).strip()
        if not text:
            raise OllamaError("Ollama returned empty response")
        return text
