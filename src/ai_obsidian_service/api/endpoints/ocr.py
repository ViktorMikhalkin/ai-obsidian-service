"""OCR preprocessing endpoints."""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from ai_obsidian_service.api.dependencies import _ocr_lock, log_structured, logger

try:
    from ai_obsidian_service.utils.ocr_preprocessing import (
        find_scanned_pdfs,
        process_pdf_with_ocr,
    )

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

router = APIRouter()


@router.post("/scan")
async def ocr_scan_directory(root: str = Body(..., embed=True)):
    """Scan directory for PDFs that need OCR (dry run)."""
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail={"code": "OCR_NOT_AVAILABLE", "message": "OCR module not installed"},
        )

    if not root:
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_ROOT", "message": "Provide 'root' field"},
        )

    p = Path(root)
    if not p.exists():
        log_structured(
            "warning", "ocr_scan_invalid_path", root=root, reason="not_found"
        )
        raise HTTPException(
            status_code=400,
            detail={"code": "ROOT_NOT_FOUND", "message": f"Path not found: {root}"},
        )
    if not p.is_dir():
        log_structured(
            "warning", "ocr_scan_invalid_path", root=root, reason="not_directory"
        )
        raise HTTPException(
            status_code=400,
            detail={"code": "ROOT_NOT_DIR", "message": f"Not a directory: {root}"},
        )

    try:
        scanned_pdfs = list(find_scanned_pdfs(p))

        log_structured(
            "info", "ocr_scan_completed", root=root, scanned_count=len(scanned_pdfs)
        )

        return {
            "root": root,
            "scanned_pdfs": [str(pdf.relative_to(p)) for pdf in scanned_pdfs],
            "count": len(scanned_pdfs),
            "scanned_at": datetime.utcnow().isoformat() + "Z",
        }
    except Exception as e:
        log_structured("error", "ocr_scan_failed", root=root, error=str(e))
        logger.exception("OCR scan failed")
        raise HTTPException(
            status_code=500,
            detail={"code": "SCAN_FAILED", "message": str(e)},
        ) from e


@router.post("/process")
async def ocr_process_directory(
    root: str = Body(...),
    workers: int = Body(4),
    language: str = Body("eng+rus+ukr"),
):
    """Process all scanned PDFs in directory with OCR (with SSE progress)."""
    if not OCR_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail={"code": "OCR_NOT_AVAILABLE", "message": "OCR module not installed"},
        )

    if not root:
        raise HTTPException(
            status_code=422,
            detail={"code": "MISSING_ROOT", "message": "Provide 'root' field"},
        )

    p = Path(root)
    if not p.exists() or not p.is_dir():
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_PATH", "message": f"Invalid directory: {root}"},
        )

    if _ocr_lock.locked():
        log_structured("warning", "ocr_process_rejected", reason="already_running")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "code": "OCR_IN_PROGRESS",
                "message": "OCR processing already in progress",
            },
        )

    async def progress_stream():
        """Generate SSE progress events for OCR processing."""
        try:
            async with _ocr_lock:
                log_structured(
                    "info", "ocr_processing_started", root=root, workers=workers
                )

                yield f"data: {json.dumps({'status': 'scanning', 'message': 'Scanning for scanned PDFs...'})}\n\n"

                scanned_pdfs = list(find_scanned_pdfs(p))
                total = len(scanned_pdfs)

                if total == 0:
                    log_structured(
                        "info",
                        "ocr_processing_completed",
                        processed=0,
                        reason="no_scanned_pdfs",
                    )
                    yield f"data: {json.dumps({'status': 'complete', 'processed': 0, 'total': 0, 'message': 'No scanned PDFs found'})}\n\n"
                    return

                log_structured("info", "scanned_pdfs_found", count=total)
                yield f"data: {json.dumps({'status': 'started', 'total': total, 'message': f'Found {total} scanned PDFs'})}\n\n"

                processed = 0
                failed = 0
                skipped = 0
                start_time = time.time()
                last_update = start_time

                async def process_single_pdf(pdf_path: Path) -> tuple[Path, bool, str]:
                    """Process single PDF, return (path, success, status)."""
                    try:
                        result = await asyncio.to_thread(
                            process_pdf_with_ocr,
                            pdf_path,
                            language=language,
                        )

                        if result is None:
                            return (pdf_path, False, "failed")
                        elif result == pdf_path:
                            return (pdf_path, True, "skipped")
                        else:
                            return (pdf_path, True, "processed")
                    except Exception as e:
                        log_structured(
                            "error", "ocr_pdf_failed", file=str(pdf_path), error=str(e)
                        )
                        return (pdf_path, False, "error")

                batch_size = min(workers, 4)

                for i in range(0, len(scanned_pdfs), batch_size):
                    batch = scanned_pdfs[i : i + batch_size]

                    results = await asyncio.gather(
                        *[process_single_pdf(pdf) for pdf in batch]
                    )

                    for _pdf_path, _success, status_type in results:
                        if status_type == "processed":
                            processed += 1
                        elif status_type == "skipped":
                            skipped += 1
                        elif status_type in ("failed", "error"):
                            failed += 1

                    total_done = processed + failed + skipped
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rate = total_done / elapsed if elapsed > 0 else 0
                    eta_seconds = (total - total_done) / rate if rate > 0 else 0

                    progress_data = {
                        "status": "processing",
                        "total": total,
                        "processed": processed,
                        "failed": failed,
                        "skipped": skipped,
                        "percent": int((total_done / total) * 100),
                        "rate": round(rate, 2),
                        "eta_seconds": int(eta_seconds),
                        "current_file": batch[-1].name if batch else "",
                    }
                    yield f"data: {json.dumps(progress_data)}\n\n"
                    last_update = current_time

                    if time.time() - last_update > 15:
                        yield ": keep-alive\n\n"
                        last_update = time.time()

                total_time = time.time() - start_time
                log_structured(
                    "info",
                    "ocr_processing_completed",
                    processed=processed,
                    failed=failed,
                    skipped=skipped,
                    duration_s=round(total_time, 2),
                )

                final_data = {
                    "status": "complete",
                    "total": total,
                    "processed": processed,
                    "failed": failed,
                    "skipped": skipped,
                    "total_time_seconds": round(total_time, 2),
                    "message": f"OCR complete: {processed} processed, {failed} failed, {skipped} skipped",
                }
                yield f"data: {json.dumps(final_data)}\n\n"

        except asyncio.CancelledError:
            log_structured("info", "ocr_processing_cancelled")
            yield f"data: {json.dumps({'status': 'cancelled', 'message': 'OCR cancelled'})}\n\n"
            raise
        except Exception as e:
            log_structured("error", "ocr_processing_failed", error=str(e))
            logger.exception("OCR processing failed")
            yield f"data: {json.dumps({'status': 'failed', 'error': str(e)})}\n\n"

    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/status")
def ocr_status():
    """Get OCR processing status."""
    ocr_available = False
    if OCR_AVAILABLE:
        try:
            result = subprocess.run(
                ["ocrmypdf", "--version"],
                capture_output=True,
                timeout=2,
            )
            ocr_available = result.returncode == 0
        except Exception:
            ocr_available = False

    return {
        "ocr_running": _ocr_lock.locked(),
        "ocr_available": ocr_available,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
