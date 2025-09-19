import importlib
import os
import sys
from collections.abc import Callable
from typing import cast

from fastapi import FastAPI

type ASGIApp = FastAPI | Callable[..., object]


def _resolve_app() -> ASGIApp | None:
    spec = os.getenv("APP_MODULE")
    candidates = [spec] if spec else []
    candidates += [
        "ai_obsidian_service.indexer.app:app",
        "ai_obsidian_service.app:app",
        "ai_obsidian_service.api.app:app",
        "ai_obsidian_service.service.app:app",
        "ai_obsidian_service.server:app",
        "ai_obsidian_service.application:app",
    ]
    for cand in candidates:
        try:
            mod_name, _, attr = cand.partition(":")
            mod = importlib.import_module(mod_name)
            obj = getattr(mod, attr or "app", None)
            if callable(obj):
                return cast(ASGIApp, obj)
        except Exception:
            continue
    return None


def main() -> int:
    app = _resolve_app()
    if app is not None:
        try:
            import uvicorn
        except ImportError:
            print("Install uvicorn to run the ASGI app", file=sys.stderr)
            return 2
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        uvicorn.run(app, host=host, port=port)
        return 0
    print("AI Obsidian Service started (no ASGI app found).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
