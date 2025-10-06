import pytest

try:
    from hypothesis import given
    from hypothesis import strategies as st
except Exception:
    pytest.skip("hypothesis not installed", allow_module_level=True)

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser


@given(st.text(min_size=1, max_size=2000))
def test_md_parser_never_crashes(s: str) -> None:
    p = MarkdownParser()
    doc = p.parse_text(s)
    assert doc is not None
    assert isinstance(doc.text, str)
