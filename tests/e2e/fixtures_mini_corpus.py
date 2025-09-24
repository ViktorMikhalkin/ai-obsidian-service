from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import pytest


def _write(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def create_md_corpus(root: Path) -> List[Path]:
    """Always-available tiny markdown corpus."""
    docs = []
    docs.append(_write(root / "intro.md", """# Intro
AI Obsidian is a tiny RAG service.

## Goals
Search, Answer, Index."""))

    docs.append(_write(root / "usage.md", """# Usage
Run /index to ingest docs.
Then /search to retrieve chunks.

## Tips
Use VECTOR_STORE_BACKEND to select memory|faiss."""))

    docs.append(_write(root / "faq.md", """# FAQ
Q: Does it persist?
A: With FAISS + VECTOR_INDEX_DIR on shutdown/startup."""))
    return docs


def create_epub_if_possible(root: Path) -> Optional[Path]:
    """Create minimal EPUB if ebooklib is installed. Otherwise return None."""
    try:
        from ebooklib import epub  # type: ignore
    except Exception:
        return None

    book = epub.EpubBook()
    book.set_identifier("mini-epub")
    book.set_title("Mini EPUB")
    book.set_language("en")

    c1 = epub.EpubHtml(title="Intro", file_name="intro.xhtml", lang="en")
    c1.set_content("<h1>Intro</h1><p>EPUB page one.</p>")
    c2 = epub.EpubHtml(title="Usage", file_name="usage.xhtml", lang="en")
    c2.set_content("<h1>Usage</h1><p>EPUB page two.</p>")

    book.add_item(c1)
    book.add_item(c2)
    book.toc = (c1, c2)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = ["nav", c1, c2]
    out = root / "mini.epub"
    epub.write_epub(str(out), book, {})
    return out


def create_pdf_if_possible(root: Path) -> Optional[Path]:
    """Create minimal PDF if reportlab is installed. Otherwise return None."""
    try:
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        return None
    out = root / "mini.pdf"
    c = canvas.Canvas(str(out))
    c.drawString(100, 750, "Mini PDF")
    c.drawString(100, 735, "E2E corpus page")
    c.save()
    return out


@pytest.fixture(scope="function")
def mini_corpus(tmp_path: Path) -> Tuple[List[Path], List[Path]]:
    """
    Returns (required_docs, optional_docs). Required are markdown files.
    Optional may include epub/pdf if libs are present.
    """
    required = create_md_corpus(tmp_path)
    optional: List[Path] = []
    e = create_epub_if_possible(tmp_path)
    if e:
        optional.append(e)
    p = create_pdf_if_possible(tmp_path)
    if p:
        optional.append(p)
    return required, optional
