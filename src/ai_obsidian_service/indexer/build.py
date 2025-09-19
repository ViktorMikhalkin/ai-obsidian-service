from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from .index_admin import rebuild_index  # real implementation
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"[build] entrypoint not found in index_admin: {e}") from e

    # Optional timeout via env (cast to int if provided)
    timeout_env = os.getenv("AIOBS_TIMEOUT_SEC")
    if timeout_env:
        try:
            timeout = int(timeout_env)
            rc = rebuild_index(timeout)  # positional int
        except ValueError:
            # Bad env value; fall back to no-arg call
            rc = rebuild_index()
    else:
        rc = rebuild_index()  # no-arg call

    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
