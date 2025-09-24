
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given
from hypothesis import strategies as st

from ai_obsidian_service.adapters.parsers.md_parser import MarkdownParser


@given(st.text(min_size=1, max_size=2000))
def test_md_parser_never_crashes(s: str) -> None:
    p = MarkdownParser()
    # if parser doesn't have parse_text, fallback to writing a tmp file
    try:
        doc = p.parse_text(s)  # type: ignore[attr-defined]
    except Exception:
        from pathlib import Path
        tmp = Path("tmp_test.md")
        tmp.write_text(s, encoding="utf-8")
        doc = p.parse(str(tmp))
        tmp.unlink(missing_ok=True)
    assert doc is not None
    assert doc.text is not None
