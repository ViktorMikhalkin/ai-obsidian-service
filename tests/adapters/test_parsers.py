from pathlib import Path

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser


def test_markdown_parser_basic(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# Title\ncontent", encoding="utf-8")
    parser = MarkdownParser()
    assert parser.can_parse(str(p))
    doc = parser.parse(str(p))
    assert doc.text.startswith("# Title")
    assert doc.mime.startswith("text/")
