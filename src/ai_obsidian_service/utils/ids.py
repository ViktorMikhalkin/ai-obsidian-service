
from __future__ import annotations

import hashlib


def doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
def source_id(rel_posix: str) -> str:
    return hashlib.sha1(rel_posix.encode('utf-8')).hexdigest()
def chunk_id(source_id_str: str, order: int) -> str:
    return f"{source_id_str}:{order}"
