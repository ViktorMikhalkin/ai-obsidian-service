"""
OCR preprocessing for scanned PDFs.

Strategy: Run ocrmypdf as a batch job before indexing.
This keeps the main indexing pipeline fast and clean.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # PyMuPDF - only for type checking

log = logging.getLogger(__name__)


def has_text_content(pdf_path: Path) -> bool:
    """
    Check if PDF has extractable text.

    Returns:
        True if PDF has text, False if it's a scanned image
    """
    try:
        import fitz  # Lazy import - only when actually checking PDFs

        doc = fitz.open(str(pdf_path))

        # Check first 3 pages (or all if fewer)
        pages_to_check = min(3, len(doc))

        for page_num in range(pages_to_check):
            page = doc[page_num]
            text = page.get_text("text")

            # If any page has substantial text, consider it text-based
            if text and len(text.strip()) > 50:
                doc.close()
                return True

        doc.close()
        return False

    except Exception as e:
        log.warning(f"Error checking PDF {pdf_path}: {e}")
        return False


def process_pdf_with_ocr(
    input_path: Path,
    output_dir: Path | None = None,
    *,
    force: bool = False,
    language: str = "eng+rus+ukr",
    skip_existing: bool = True,
) -> Path | None:
    """
    Process a PDF with OCR if needed.

    Args:
        input_path: Path to input PDF
        output_dir: Directory for OCR output (default: same as input)
        force: Force OCR even if PDF has text
        language: Tesseract language codes (e.g., "eng+rus+ukr")
        skip_existing: Skip if output already exists

    Returns:
        Path to processed PDF, or None if skipped/failed
    """
    if output_dir is None:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_ocr.pdf"

    # Skip if already processed
    if skip_existing and output_path.exists():
        log.info(f"Skipping {input_path.name}: OCR output already exists")
        return output_path

    # Check if OCR is needed
    if not force and has_text_content(input_path):
        log.info(f"Skipping {input_path.name}: Already has text content")
        return input_path

    # Run ocrmypdf
    try:
        log.info(f"Processing {input_path.name} with OCR...")

        cmd = [
            "ocrmypdf",
            "--language",
            language,
            "--force-ocr" if force else "--skip-text",
            "--optimize",
            "1",
            "--fast-web-view",
            "1",
            "--deskew",  # Fix rotated scans
            "--clean",  # Clean up artifacts
            "--quiet",  # Suppress progress output
            str(input_path),
            str(output_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes per PDF
        )

        if result.returncode == 0:
            log.info(f"OCR complete: {output_path.name}")
            return output_path
        else:
            log.error(f"OCR failed for {input_path.name}: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        log.error(f"OCR timeout for {input_path.name}")
        return None

    except FileNotFoundError:
        log.error("ocrmypdf not found. Install: pip install ocrmypdf")
        return None

    except Exception as e:
        log.error(f"OCR error for {input_path.name}: {e}")
        return None


def find_scanned_pdfs(directory: Path) -> Iterator[Path]:
    """
    Find all scanned PDFs (without text) in a directory.

    Yields:
        Paths to scanned PDFs
    """
    for pdf_path in directory.rglob("*.pdf"):
        if not pdf_path.is_file():
            continue

        # Skip already processed files
        if pdf_path.stem.endswith("_ocr"):
            continue

        if not has_text_content(pdf_path):
            yield pdf_path


def batch_process_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    *,
    max_workers: int = 4,
    language: str = "eng+rus+ukr",
    dry_run: bool = False,
) -> dict[str, int]:
    """
    Batch process all scanned PDFs in a directory.

    Args:
        input_dir: Directory to scan for PDFs
        output_dir: Directory for OCR output (default: same as input)
        max_workers: Number of parallel OCR processes
        language: Tesseract language codes
        dry_run: Just detect scanned PDFs without processing

    Returns:
        Statistics: {scanned, processed, failed, skipped}
    """
    log.info(f"Scanning directory: {input_dir}")

    scanned_pdfs = list(find_scanned_pdfs(input_dir))

    stats = {
        "scanned": len(scanned_pdfs),
        "processed": 0,
        "failed": 0,
        "skipped": 0,
    }

    if not scanned_pdfs:
        log.info("No scanned PDFs found")
        return stats

    log.info(f"Found {len(scanned_pdfs)} scanned PDFs")

    if dry_run:
        for pdf_path in scanned_pdfs:
            print(f"  - {pdf_path.relative_to(input_dir)}")
        return stats

    # Process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_pdf_with_ocr,
                pdf_path,
                output_dir,
                language=language,
            ): pdf_path
            for pdf_path in scanned_pdfs
        }

        for future in as_completed(futures):
            pdf_path = futures[future]

            try:
                result = future.result()

                if result is not None:
                    if result == pdf_path:
                        stats["skipped"] += 1
                    else:
                        stats["processed"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                log.error(f"Error processing {pdf_path.name}: {e}")
                stats["failed"] += 1

    log.info(
        f"OCR batch complete: {stats['processed']} processed, "
        f"{stats['failed']} failed, {stats['skipped']} skipped"
    )

    return stats


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCR preprocessing for scanned PDFs")
    parser.add_argument("input_dir", type=Path, help="Directory containing PDFs")
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory (default: same as input)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel OCR processes (default: 4)",
    )
    parser.add_argument(
        "--language",
        default="eng+rus+ukr",
        help="Tesseract language codes (default: eng+rus+ukr)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just list scanned PDFs without processing",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Run batch processing
    stats = batch_process_directory(
        args.input_dir,
        args.output_dir,
        max_workers=args.workers,
        language=args.language,
        dry_run=args.dry_run,
    )

    print("\nResults:")
    print(f"  Scanned PDFs found: {stats['scanned']}")
    print(f"  Processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
