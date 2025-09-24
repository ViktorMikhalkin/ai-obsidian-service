
from __future__ import annotations

from pathlib import Path

from ai_obsidian_service.utils.ids import chunk_id, doc_hash, source_id
from ai_obsidian_service.utils.paths import collection_of, norm_rel


def test_source_and_chunk_id_stability() -> None:
    rel = "Area/Topic/note.md"
    sid = source_id(rel)
    assert len(sid) == 40
    cid0 = chunk_id(sid, 0)
    cid1 = chunk_id(sid, 1)
    assert cid0 != cid1 and cid0.startswith(sid + ":")
    h = doc_hash("hello")
    assert len(h) == 64

def test_norm_rel_and_collection() -> None:
    root = Path("/vault")
    p = Path("/vault/Area/Topic/note.md")
    rel = norm_rel(root, p)
    assert rel == "Area/Topic/note.md"
    assert collection_of(rel) == "Area/Topic"
    assert collection_of("note.md") == ""
