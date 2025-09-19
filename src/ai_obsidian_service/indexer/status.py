# Thin shim to provide a stable entrypoint for
#   python -m ai_obsidian_service.indexer.status
from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        from .index_admin import get_index_stats  # ✅ correct name
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"[status] entrypoint not found in index_admin: {e}") from e

    stats = get_index_stats()  # обычно без аргументов
    try:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    except Exception:
        print(stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
