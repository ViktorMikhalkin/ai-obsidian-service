from __future__ import annotations

import io
from pathlib import Path

import pytest

from ai_obsidian_service.adapters.services.search_service import SearchService

pytestmark = pytest.mark.e2e


def _minimal_pdf_bytes(text: str) -> bytes:
    """
    Build a tiny 1-page PDF with 'text' drawn using Helvetica.
    No compression, valid xref, works with pypdf extract_text().
    """
    # content stream (plain text operators)
    content = f"""BT
/F1 24 Tf
1 0 0 1 72 720 Tm
({text}) Tj
ET
""".encode("latin-1")

    len_content = len(content)

    parts: list[bytes] = []
    parts.append(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    # 1: Catalog
    parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    # 2: Pages
    parts.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    # 3: Page
    parts.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>\n"
        b"endobj\n"
    )
    # 4: Font
    parts.append(b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    # 5: Contents
    parts.append(
        f"5 0 obj\n<< /Length {len_content} >>\nstream\n".encode("latin-1")
        + content
        + b"endstream\nendobj\n"
    )

    # xref
    # body = b"".join(parts)
    # xref_offset = len(body)
    # object byte offsets (rough but works because we build sequentially)
    # We need exact offsets — so rebuild with measured offsets.
    objs = [
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents 5 0 R >>\n"
            b"endobj\n"
        ),
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        f"5 0 obj\n<< /Length {len_content} >>\nstream\n".encode("latin-1")
        + content
        + b"endstream\nendobj\n",
        ]
    # recompute offsets precisely
    offsets = []
    cursor = 0
    for chunk in objs:
        offsets.append(cursor)
        cursor += len(chunk)

    xref = io.BytesIO()
    xref.write(f"xref\n0 {len(objs)+1}\n".encode("latin-1"))
    # obj 0 (free)
    xref.write(b"0000000000 65535 f \n")
    # objs 1..5
    for off in offsets[1:]:
        xref.write(f"{off:010d} 00000 n \n".encode("latin-1"))

    trailer = (
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n"
            + f"{cursor}".encode("latin-1")
            + b"\n%%EOF\n"
    )
    return b"".join(objs) + xref.getvalue() + trailer


def _make_epub(path: Path, text: str) -> None:
    """
    Create a tiny EPUB with one chapter containing 'text'.
    Requires ebooklib.
    """
    try:
        from ebooklib import epub  # type: ignore
    except Exception:
        pytest.skip("ebooklib is not available; skipping EPUB part of the e2e test")

    book = epub.EpubBook()
    book.set_title("Test EPUB")
    book.add_author("AI Obsidian Service")

    chapter = epub.EpubHtml(title="Chapter 1", file_name="chap_1.xhtml", lang="en")
    chapter.content = f"<html><body><h1>Chapter</h1><p>{text}</p></body></html>"
    book.add_item(chapter)
    book.spine = ["nav", chapter]
    book.toc = (chapter,)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    epub.write_epub(str(path), book)  # type: ignore


@pytest.fixture
def corpus_with_md_pdf_epub(tmp_path: Path) -> Path:
    """
    Build a tiny corpus with:
      - notes/A.md (contains both tokens)
      - docs/X.pdf (contains 'pdfonlytoken')
      - library/Y.epub (contains 'epubonlytoken')
    """
    notes = tmp_path / "notes"
    docs = tmp_path / "docs"
    lib = tmp_path / "library"
    notes.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    lib.mkdir(parents=True, exist_ok=True)

    (notes / "A.md").write_text(
        "# A\n\nmdtoken pdfonlytoken epubonlytoken\n", encoding="utf-8"
    )

    # PDF
    pdf_bytes = _minimal_pdf_bytes("pdfonlytoken")
    (docs / "X.pdf").write_bytes(pdf_bytes)

    # EPUB
    _make_epub(lib / "Y.epub", "epubonlytoken")

    return tmp_path


def test_mix_parsers_index_and_search(
        search_service: SearchService, corpus_with_md_pdf_epub: Path
):
    # Index all supported files seen under vault
    for p in sorted(corpus_with_md_pdf_epub.rglob("*")):
        if p.suffix.lower() in {".md", ".pdf", ".epub"}:
            search_service.index_path(str(p))

    # Search token that exists only in PDF
    res_pdf = search_service.search_text("pdfonlytoken", top_k=5)
    assert res_pdf.hits, "Expected hits from PDF"
    # Search token that exists only in EPUB
    res_epub = search_service.search_text("epubonlytoken", top_k=5)
    assert res_epub.hits, "Expected hits from EPUB"
    # And a token present in MD (sanity)
    res_md = search_service.search_text("mdtoken", top_k=5)
    assert res_md.hits, "Expected hits from MD"
