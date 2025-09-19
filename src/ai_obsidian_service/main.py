"""
Main entry point for AI Obsidian Service.
Handles dynamic ASGI app resolution and runs the server.
"""

import importlib
import logging
import os
import sys
from collections.abc import Callable

from fastapi import FastAPI

type ASGIApp = FastAPI | Callable[..., object]


class AppNotFoundException(Exception):
    pass


class UvicornNotInstalledException(Exception):
    pass


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def resolve_app(candidates: list[str] | None = None) -> ASGIApp:
    """
    Resolve and import the ASGI application from a list of candidates.
    """
    if candidates is None:
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
                logging.info(f"ASGI app found: {cand}")
                return obj
        except Exception as e:
            logging.debug(f"Failed to import {cand}: {e}")
    raise AppNotFoundException("No valid ASGI app found in candidates.")


def run_app(app: ASGIApp):
    """
    Run the ASGI app using uvicorn.
    """
    try:
        import uvicorn
    except ImportError as e:
        raise UvicornNotInstalledException(
            "Install uvicorn to run the ASGI app."
        ) from e
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    logging.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


def main() -> int:
    setup_logging()
    try:
        app = resolve_app()
        run_app(app)
        return 0
    except AppNotFoundException as e:
        logging.error(str(e))
        print("AI Obsidian Service started (no ASGI app found).")
        return 1
    except UvicornNotInstalledException as e:
        logging.error(str(e))
        print(str(e), file=sys.stderr)
        return 2
    except Exception:
        logging.exception("Unexpected error starting the service.")
        return 3


if __name__ == "__main__":
    sys.exit(main())
