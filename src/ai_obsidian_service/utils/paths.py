
from __future__ import annotations

from pathlib import Path, PurePosixPath


def norm_rel(vault_root: Path, abs_path: Path) -> str:
    rel = abs_path.resolve().relative_to(vault_root.resolve())
    return PurePosixPath(rel).as_posix()
def collection_of(rel_posix: str) -> str:
    parts = rel_posix.split('/')
    return '/'.join(parts[:-1]) if len(parts) > 1 else ''
