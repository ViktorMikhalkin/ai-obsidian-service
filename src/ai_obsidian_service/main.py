from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ai_obsidian_service.adapters.services.search_service import SearchService

# Expose FastAPI app for tests (and for `uvicorn ai_obsidian_service.main:app` if desired)
from ai_obsidian_service.api.app import app  # noqa: F401

# NEW: imports for typed search response
from ai_obsidian_service.api.mappers import hits_to_search_response
from ai_obsidian_service.api.schemas import SearchResponse
from ai_obsidian_service.config.container import (
    build_index_corpus,
    build_search_service,
)
from ai_obsidian_service.domain.models import Query


def _to_plain(obj: Any) -> Any:
    """Best-effort JSON-serializable view for SearchResult/Hit/etc."""
    if is_dataclass(obj):
        # Fix: Check that obj is an instance, not a class
        if not isinstance(obj, type):
            return asdict(obj)
        else:
            return str(obj)
    if hasattr(obj, "__dict__"):
        safe: dict[str, Any] = {}
        for k, v in obj.__dict__.items():
            try:
                json.dumps(v)
                safe[k] = v
            except Exception:
                safe[k] = str(v)
        return safe
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _print_json(payload: Any) -> None:
    print(json.dumps(_to_plain(payload), ensure_ascii=False, indent=2))


def _print_pydantic(model: SearchResponse) -> None:
    """Dump a pydantic model (v1 or v2) as pretty JSON."""
    try:
        data = model.model_dump()  # pydantic v2
    except Exception:
        try:
            data = model.dict()  # pydantic v1
        except Exception:
            data = _to_plain(model)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _shutdown_service(service: SearchService | None) -> None:
    """Properly shutdown service and its components."""
    if service is None:
        return

    try:
        # Try to shutdown the store if it has a close method
        if hasattr(service, "index") and hasattr(service.index, "store") and hasattr(service.index.store, "close"):
            service.index.store.close()
    except Exception:
        pass

    try:
        # Try to close the index if it has a close method
        if hasattr(service, "index") and hasattr(service.index, "close"):
            service.index.close()
    except Exception:
        pass


def _install_sigterm(service: SearchService | None) -> None:
    def _handler(_sig: int, _frame) -> None:
        _shutdown_service(service)
        sys.exit(0)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _handler)


# ---------------------- commands ---------------------- #

def cmd_index_root(args: argparse.Namespace) -> int:
    """
    Bulk indexing of a directory: selects parsers, parses and delegates indexing to service.
    """
    root = Path(args.root)
    if not root.exists():
        print(f"[index] root not found: {root}", file=sys.stderr)
        return 2

    usecase = build_index_corpus(index_dir=args.index_dir)
    count = usecase.run(str(root))
    print(f"[index] indexed files: {count}")
    return 0


def cmd_index_file(args: argparse.Namespace) -> int:
    """
    Index a single file (convenient for incremental indexing).
    """
    path = Path(args.path)
    if not path.exists():
        print(f"[index-file] not found: {path}", file=sys.stderr)
        return 2

    service = build_search_service(index_dir=args.index_dir)
    try:
        _install_sigterm(service)
        chunks = service.index_path(str(path))
        print(f"[index-file] chunks: {chunks}  path: {path}")
        return 0
    finally:
        _shutdown_service(service)


def cmd_search(args: argparse.Namespace) -> int:
    """
    Search by text query. Prints JSON result.
    """
    service = build_search_service(index_dir=args.index_dir)
    try:
        _install_sigterm(service)
        result = service.search_text(args.query, top_k=args.top_k)

        # Build a typed API response; do NOT assign a dict back to `result`.
        # Also, pass `resolve_meta` correctly (it expects chunk_id: str).
        def resolve_meta_wrapper(chunk_id: str) -> dict[str, Any]:
            return service.resolve_meta(chunk_id=chunk_id)

        resp: SearchResponse = hits_to_search_response(
            Query(text=args.query, top_k=args.top_k),
            result.hits,
            resolve_meta_wrapper,
        )
        _print_pydantic(resp)
        return 0
    finally:
        _shutdown_service(service)


def cmd_status(_args: argparse.Namespace) -> int:
    """
    Quick status from on-disk metadata (if you maintain JSONL alongside the index).
    """
    idx = Path("index/index.jsonl")
    n = sum(1 for _ in idx.open()) if idx.exists() else 0
    print(f"[status] chunks: {n}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """
    Start API (FastAPI/Uvicorn) in dev mode.
    """
    try:
        import uvicorn
    except Exception as e:
        print(f"[serve] uvicorn not installed: {e}", file=sys.stderr)
        return 2

    host = args.host or "127.0.0.1"
    port = int(args.port or 8000)
    reload = bool(args.reload)

    # FastAPI app path unified here:
    uvicorn.run("ai_obsidian_service.api.app:app", host=host, port=port, reload=reload)
    return 0


# ---------------------- CLI wiring ---------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ai-obsidian",
        description="AI→Obsidian service entrypoint",
    )
    p.add_argument("--index-dir", default=None, help="Directory for FAISS index (optional)")

    sp = p.add_subparsers(dest="cmd", required=True)

    # index-root
    p_idx = sp.add_parser("index", help="Bulk-index a directory recursively")
    p_idx.add_argument("root", help="Root folder to scan")
    p_idx.set_defaults(func=cmd_index_root)

    # index-file
    p_file = sp.add_parser("index-file", help="Index a single file (incremental)")
    p_file.add_argument("path", help="Path to a file")
    p_file.set_defaults(func=cmd_index_file)

    # search
    p_search = sp.add_parser("search", help="Search by a text query")
    p_search.add_argument("query", help="Query text")
    p_search.add_argument("--top-k", type=int, default=5, help="Number of hits to return")
    p_search.set_defaults(func=cmd_search)

    # status
    p_stat = sp.add_parser("status", help="Quick status by on-disk JSONL metadata")
    p_stat.set_defaults(func=cmd_status)

    # serve
    p_srv = sp.add_parser("serve", help="Run FastAPI server (uvicorn)")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)
    p_srv.add_argument("--reload", action="store_true", help="Dev autoreload")
    p_srv.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
