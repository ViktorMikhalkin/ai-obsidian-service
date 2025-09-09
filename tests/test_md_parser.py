from pathlib import Path
from indexer.parsers.md_parser import parse_markdown

def test_md_parser(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text("""---
title: T
---
# Header
Some *text* with [link](http://x).
""", encoding="utf-8")
    parsed = parse_markdown(p)
    assert parsed is not None
    assert "Some" in parsed.text
