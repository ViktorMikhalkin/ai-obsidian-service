from hypothesis import given, strategies as st

from ai_obsidian_service.adapters.parsers.md_parser import parse_markdown


text_no_ctrl = st.text(alphabet=st.characters(blacklist_categories=("Cc",)), min_size=0, max_size=200)

@given(text_no_ctrl)
def test_no_crash_and_sections_nonempty(text):
    # не падаем и возвращаем хотя бы 1 секцию
    sections = parse_markdown(text)
    assert isinstance(sections, list)
    assert len(sections) >= 1
    for title, body in sections:
        assert isinstance(title, str)
        assert isinstance(body, str)

@given(st.lists(st.sampled_from(["# A", "## B", "### C", "", "text", "   ", "— unicode —"]), min_size=0, max_size=40))
def test_headers_and_blanks_do_not_lose_text(tokens):
    text = "\n".join(tokens)
    sections = parse_markdown(text)
    # все не-заголовочные строки должны встречаться в телах секций
    bodies = "\n\n".join(body for _, body in sections)
    for t in tokens:
        if not t.startswith("#"):
            assert t.strip() == "" or t in bodies

@given(st.lists(st.text(min_size=0, max_size=30), min_size=0, max_size=20))
def test_sections_never_lose_content(lines):
    text = "\n".join(lines)
    sections = parse_markdown(text)
    # Re-join bodies must contain all non-header lines in order (weak invariant)
    bodies = [b for (_t, b) in sections if b]
    flattened = "\n\n".join(bodies)
    # We don't demand exact equality because headers are cut out;
    # but every non-header line must appear in the flattened bodies.
    for ln in lines:
        if not ln.startswith("#"):
            assert ln in flattened
