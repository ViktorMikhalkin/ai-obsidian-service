import os

from ollama import Client


def _client():
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    return Client(host=host)


def ollama_generate(prompt: str, model: str = "qwen2.5:7b-instruct") -> str:
    resp = _client().generate(model=model, prompt=prompt)
    return str(resp.get("response", ""))
