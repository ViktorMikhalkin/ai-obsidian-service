# tests/service/test_search_service.py
from __future__ import annotations

import os

from ai_obsidian_service.domain.models import Query


def test_collection_filter_works(search_service):
    """
    The mini_vault has:
      - note1.md at the root (collection = "" or None)
      - sub/note2.md under 'sub' (collection = "sub")

    We check that filtering by collection="sub" yields hits only from that folder.
    """
    # Ensure unit backend
    assert os.environ.get("VECTOR_STORE_BACKEND") == "memory"

    # Query that matches both notes
    q = Query("hello", top_k=5)
    res_all = search_service.search_text(q.text, top_k=q.top_k)
    assert res_all.hits, "expected some hits without collection filter"

    # Now filter by collection 'sub'
    res_sub = search_service.search_text(q.text, top_k=5, collection="sub")
    assert res_sub.hits, "expected hits when filtering by collection='sub'"

    # All returned hits must point to files under sub/
    for h in res_sub.hits:
        path = None
        if getattr(h, "chunk", None) is not None and getattr(h.chunk, "metadata", None):
            path = h.chunk.metadata.get("path")
        else:
            meta = getattr(h, "metadata", None)
            if meta:
                path = meta.get("path")

        assert path is None or "/sub/" in path or str(path).startswith("sub/"), f"unexpected path: {path}"

    # Ensure no root note is in the filtered set
    root_hits = [
        h for h in res_sub.hits
        if getattr(h, "chunk", None)
           and getattr(h.chunk, "metadata", None)
           and str(h.chunk.metadata.get("path", "")).endswith("note1.md")
    ]
    assert not root_hits, "collection='sub' must exclude root note hits"
