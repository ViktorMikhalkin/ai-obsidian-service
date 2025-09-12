from pathlib import Path
from typing import Iterator, Tuple
import subprocess
import shutil


def extract_pdf_per_pages(pdf_path: Path) -> Iterator[Tuple[int, str]]:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("Missing `pdftotext`. Install poppler-utils.")
    cmd = ["pdftotext", "-layout", "-q", str(pdf_path), "-"]
    out = subprocess.check_output(cmd, text=True, errors="ignore")
    pages = out.split("\f")
    for i, page in enumerate(pages, start=1):
        text = page.strip()
        if text:
            yield i, text
