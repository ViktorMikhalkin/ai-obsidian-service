from pathlib import Path

from ai_obsidian_service.adapters.parsers.md_parser import (
    MarkdownParser,
    parse_markdown,
)


def test_parse_markdown_function():
    """Test the parse_markdown function directly."""
    text = """# Header 1
Content under header 1.

## Header 2
Content under header 2.

Some text without header.
"""
    sections = parse_markdown(text)
    assert len(sections) == 3
    assert sections[0] == ("Header 1", "Content under header 1.")
    assert sections[1] == ("Header 2", "Content under header 2.\n\nSome text without header.")


def test_parse_markdown_with_preface():
    """Test markdown with content before first header."""
    text = """Some preface content.

# First Header
Content after header.
"""
    sections = parse_markdown(text)
    assert len(sections) == 2
    assert sections[0] == ("", "Some preface content.")
    assert sections[1] == ("First Header", "Content after header.")


def test_parse_markdown_empty():
    """Test empty markdown."""
    sections = parse_markdown("")
    assert sections == [("", "")]


def test_parse_markdown_no_headers():
    """Test markdown with no headers."""
    text = "Just some plain text\nwith multiple lines."
    sections = parse_markdown(text)
    assert len(sections) == 1
    assert sections[0] == ("", "Just some plain text\nwith multiple lines.")


def test_markdown_parser_class(tmp_path: Path):
    """Test the MarkdownParser class."""
    parser = MarkdownParser()

    # Test can_parse
    assert parser.can_parse("test.md")
    assert parser.can_parse("test.markdown")
    assert not parser.can_parse("test.txt")
    assert not parser.can_parse("test.pdf")

    # Test parse
    p = tmp_path / "test.md"
    content = """# Header
Some *text* with [link](http://x).
"""
    p.write_text(content, encoding="utf-8")

    doc = parser.parse(str(p))
    assert doc.path == str(p)
    assert doc.mime == "text/markdown"
    assert "Header" in doc.text
    assert "Some *text*" in doc.text  # Raw text, not processed


def test_markdown_parser_with_frontmatter(tmp_path: Path):
    """Test that the parser handles files with frontmatter."""
    parser = MarkdownParser()

    p = tmp_path / "with_frontmatter.md"
    content = """---
title: Test
---
# Main Header
Content here.
"""
    p.write_text(content, encoding="utf-8")

    doc = parser.parse(str(p))
    assert doc.path == str(p)
    # The parser returns raw text, so frontmatter should be included
    assert "title: Test" in doc.text
    assert "Main Header" in doc.text
