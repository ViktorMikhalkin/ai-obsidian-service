from hypothesis import given, strategies as st

from ai_obsidian_service.adapters.parsers.md_parser import parse_markdown


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
