import re
from collections.abc import Iterator

from ebooklib import epub

TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(html: str) -> str:
    text = TAG_RE.sub(" ", html)
    return re.sub(r"\s+", " ", text).strip()


def iter_epub_docs(epub_path) -> Iterator[tuple[str, str]]:
    book = epub.read_epub(str(epub_path))
    for item in book.get_items():
        if item.get_type() == 9:
            frag_id = item.get_name()
            html = item.get_content().decode(errors="ignore")
            yield frag_id, _strip_html(html)
