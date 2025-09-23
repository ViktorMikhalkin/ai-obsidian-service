from pathlib import Path

from ai_obsidian_service.adapters.parsers.md_parser import (
    MarkdownParser,
    parse_markdown,
)


def test_parse_markdown_function():
    """
    Business rule (indexing-oriented, not strict Markdown semantics):
    - Each ATX header starts a new section (title = header text).
    - A trailing/orphan block (separated by a blank line and not preceded by a header)
      becomes a separate section with an empty title "".
    """
    text = (
        "# Header 1\n"
        "Content under header 1.\n\n"
        "## Header 2\n"
        "Content under header 2.\n\n"
        "Some text without header.\n"
    )

    sections = parse_markdown(text)
    assert len(sections) == 3
    assert sections[0] == ("Header 1", "Content under header 1.")
    # Orphan text does NOT glue to the last header's body anymore:
    assert sections[1] == ("Header 2", "Content under header 2.")
    assert sections[2] == ("", "Some text without header.")


def test_parse_markdown_with_preface():
    """
    Text before the first header becomes a separate section with empty title "".
    """
    text = "Some preface content.\n\n# First Header\nContent after header.\n"
    sections = parse_markdown(text)
    assert len(sections) == 2
    assert sections[0] == ("", "Some preface content.")
    assert sections[1] == ("First Header", "Content after header.")


def test_parse_markdown_empty():
    """
    Empty document returns a single ('', '') section to avoid special-casing downstream.
    """
    sections = parse_markdown("")
    assert sections == [("", "")]


def test_parse_markdown_no_headers():
    """
    A document without any headers is a single unnamed section with the whole text as body.
    """
    text = "Just some plain text\nwith multiple lines."
    sections = parse_markdown(text)
    assert len(sections) == 1
    assert sections[0] == ("", "Just some plain text\nwith multiple lines.")


def test_markdown_parser_class(tmp_path: Path):
    """
    MarkdownParser returns raw text; sectioning is done by parse_markdown().
    """
    parser = MarkdownParser()

    # can_parse
    assert parser.can_parse("test.md")
    assert parser.can_parse("test.markdown")
    assert not parser.can_parse("test.txt")
    assert not parser.can_parse("test.pdf")

    # parse
    p = tmp_path / "test.md"
    content = "# Header\nSome *text* with [link](http://x).\n"
    p.write_text(content, encoding="utf-8")

    doc = parser.parse(str(p))
    assert doc.path == str(p)
    assert doc.mime == "text/markdown"
    assert "Header" in doc.text
    assert "Some *text*" in doc.text  # raw text, not rendered/processed


def test_markdown_parser_with_frontmatter(tmp_path: Path):
    """
    Parser keeps raw frontmatter in the text; it does not strip or parse it.
    """
    parser = MarkdownParser()

    p = tmp_path / "with_frontmatter.md"
    content = "---\ntitle: Test\n---\n# Main Header\nContent here.\n"
    p.write_text(content, encoding="utf-8")

    doc = parser.parse(str(p))
    assert doc.path == str(p)
    assert "title: Test" in doc.text  # frontmatter is kept as-is
    assert "Main Header" in doc.text


def test_multiple_orphan_blocks_between_headers():
    """
    Orphan blocks *between* headers, if separated by a blank line from the previous header’s body,
    must become individual empty-title sections to preserve navigability for indexing/search.
    """
    text = "# H1\nBody 1.\n\nOrphan A.\n\nOrphan B.\n\n## H2\nBody 2.\n"
    sections = parse_markdown(text)

    # Expected: H1 body, Orphan A, Orphan B, H2 body
    assert len(sections) == 4
    assert sections[0] == ("H1", "Body 1.")
    assert sections[1] == ("", "Orphan A.")
    assert sections[2] == ("", "Orphan B.")
    assert sections[3] == ("H2", "Body 2.")


def test_back_to_back_headers_produce_empty_bodies():
    """
    Back-to-back headers (no body lines between) still produce sections with empty bodies.
    """
    text = "# H1\n## H2\n### H3\nBody 3.\n"
    sections = parse_markdown(text)

    assert len(sections) == 3
    assert sections[0] == ("H1", "")
    assert sections[1] == ("H2", "")
    assert sections[2] == ("H3", "Body 3.")
