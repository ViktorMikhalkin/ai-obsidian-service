from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.config.container import (
    build_index_corpus,
    build_search_service,
)


def _to_plain(obj: Any) -> Any:
    """Best-effort JSON-serializable view for SearchResult/Hit/etc."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        # avoid non-serializable attributes
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


def _install_sigterm(service: SearchService | None) -> None:
    def _handler(sig: int, _frame) -> None:
        # graceful shutdown if running long-lived command (serve…)
        if service is not None:
            try:
                service.shutdown()
            except Exception:
                pass
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