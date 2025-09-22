from pathlib import Path

from ai_obsidian_service.adapters.parsers.md_parser import parse_markdown


def test_md_parser(tmp_path: Path):
    p = tmp_path / "note.md"
    p.write_text(
        """---
title: T
---
# Header
Some *text* with [link](http://x).
""",
        encoding="utf-8",
    )
    parsed = parse_markdown(p)
    assert parsed is not None
    assert "Some" in parsed.text


def test_md_parser_no_frontmatter(tmp_path: Path):
    p = tmp_path / "note2.md"
    p.write_text("# Header\nBody without frontmatter\n", encoding="utf-8")
    parsed = parse_markdown(p)
    assert parsed is not None
    assert "Body without frontmatter" in parsed.text


def test_md_parser_empty_file(tmp_path: Path):
    p = tmp_path / "empty.md"
    p.write_text("", encoding="utf-8")
    parsed = parse_markdown(p)
    # Parser should be resilient to empty input: None or empty text
    assert parsed is None or getattr(parsed, "text", "") in ("", None)
