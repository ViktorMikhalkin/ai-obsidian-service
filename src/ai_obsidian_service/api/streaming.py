"""Server-Sent Events (SSE) streaming utilities for long-running operations."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from ai_obsidian_service.config.container import build_index_corpus


async def create_rebuild_stream(
    root: str, index_dir: str | None, force: bool = False
) -> AsyncGenerator[str, None]:
    """Generate SSE progress events for index rebuild."""
    from ai_obsidian_service.api.dependencies import log_structured

    try:
        log_structured("info", "index_rebuild_started", root=root, force=force)
        yield f"data: {json.dumps({'status': 'scanning', 'message': 'Scanning directory...'})}\n\n"

        p = Path(root)
        usecase = build_index_corpus(index_dir=index_dir)
        rebuild_service = usecase.service

        # Load existing registry for incremental indexing
        if index_dir and hasattr(rebuild_service.index, "load_registry"):
            registry_path = Path(index_dir) / "doc_registry.json"
            if registry_path.exists():
                try:
                    rebuild_service.index.load_registry(registry_path)
                    log_structured(
                        "info",
                        "registry_preloaded",
                        documents=len(rebuild_service.index._doc_registry),
                    )
                except Exception as e:
                    log_structured("warning", "registry_preload_failed", error=str(e))

        file_paths = []
        for path in p.rglob("*"):
            if not path.is_file():
                continue
            spath = str(path)
            for parser in usecase.parsers:
                if parser.can_parse(spath):
                    file_paths.append(path)
                    break

        total = len(file_paths)
        log_structured("info", "files_scanned", total=total)

        if total == 0:
            log_structured(
                "info", "index_rebuild_completed", chunks=0, files=0, reason="no_files"
            )
            yield f"data: {json.dumps({'status': 'complete', 'indexed': 0, 'total': 0, 'message': 'No indexable files found'})}\n\n"
            return

        yield f"data: {json.dumps({'status': 'started', 'total': total, 'message': f'Found {total} files to index'})}\n\n"

        async def process_file(file_path: Path) -> tuple[Path, int | None, bool, bool]:
            """Process single file, return (path, chunks, was_skipped, needs_ocr)."""
            try:
                if hasattr(rebuild_service.index, "index_document") and hasattr(
                    rebuild_service.index, "needs_reindex"
                ):
                    parser = None
                    for par in usecase.parsers:
                        if par.can_parse(str(file_path)):
                            parser = par
                            break

                    if not parser:
                        log_structured(
                            "error",
                            "file_index_failed",
                            file=str(file_path),
                            reason="no_parser_found",
                        )
                        return (file_path, None, False, False)

                    # Parse document in thread
                    def parse_doc() -> Any:
                        return parser.parse(str(p), str(file_path))  # type: ignore[call-arg]

                    doc = await asyncio.to_thread(parse_doc)
                    if not doc:
                        log_structured(
                            "error",
                            "file_index_failed",
                            file=str(file_path),
                            reason="parse_returned_none",
                        )
                        return (file_path, None, False, False)

                    # Extract metadata safely
                    metadata = doc.metadata if doc.metadata is not None else {}
                    needs_ocr = bool(metadata.get("needs_ocr", False))

                    # Index document in thread
                    def index_doc() -> Any:
                        # Try with force parameter (EnhancedEmbeddingIndex)
                        try:
                            return rebuild_service.index.index_document(
                                doc, force=force
                            )  # type: ignore[call-arg]
                        except TypeError:
                            # Fall back to without force (basic EmbeddingIndex)
                            return rebuild_service.index.index_document(doc)

                    res: Any = await asyncio.to_thread(index_doc)

                    if isinstance(res, dict):
                        skipped = bool(res.get("skipped"))
                        if skipped:
                            return (file_path, 0, True, needs_ocr)
                        indexed = bool(res.get("indexed"))
                        if indexed:
                            chunks_val = int(res.get("chunks", 0))
                            return (file_path, chunks_val, False, needs_ocr)
                        else:
                            log_structured(
                                "error",
                                "file_index_failed",
                                file=str(file_path),
                                reason="indexing_returned_false",
                            )
                            return (file_path, None, False, needs_ocr)
                    elif isinstance(res, int):
                        # Fallback interface: returns chunk count
                        return (file_path, res, False, needs_ocr)
                    else:
                        log_structured(
                            "error",
                            "file_index_failed",
                            file=str(file_path),
                            reason=f"unexpected_result_type:{type(res).__name__}",
                        )
                        return (file_path, None, False, needs_ocr)

                chunks = await asyncio.to_thread(
                    rebuild_service.index_path, str(file_path)
                )
                if chunks is None:
                    log_structured(
                        "error",
                        "file_index_failed",
                        file=str(file_path),
                        reason="fallback_returned_none",
                    )
                return (file_path, chunks, False, False)

            except Exception as e:
                log_structured(
                    "error", "file_index_failed", file=str(file_path), error=str(e)
                )
                return (file_path, None, False, False)

        count = 0
        processed = 0
        errors = 0
        skipped = 0
        needs_ocr_count = 0
        ocr_files = []
        last_update = time.time()
        last_checkpoint = time.time()
        start_time = time.time()
        batch_size = 4

        checkpoint_interval = 50
        checkpoint_time_interval = 300

        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            results = await asyncio.gather(
                *[process_file(fp) for fp in batch], return_exceptions=False
            )

            for file_path, chunks, was_skipped, needs_ocr in results:
                processed += 1
                if was_skipped:
                    skipped += 1
                elif chunks is not None:
                    count += chunks
                    if needs_ocr:
                        needs_ocr_count += 1
                        ocr_files.append(str(file_path.relative_to(p)))
                else:
                    errors += 1

            current_time = time.time()
            should_update = current_time - last_update >= 2 or processed == total

            if should_update:
                elapsed = current_time - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                eta_seconds = (total - processed) / rate if rate > 0 else 0

                progress_data = {
                    "status": "indexing",
                    "processed": processed,
                    "total": total,
                    "chunks": count,
                    "errors": errors,
                    "skipped": skipped,
                    "needs_ocr": needs_ocr_count,
                    "percent": int((processed / total) * 100),
                    "rate": round(rate, 2),
                    "eta_seconds": int(eta_seconds),
                    "current_file": batch[-1].name if batch else "",
                }
                yield f"data: {json.dumps(progress_data)}\n\n"
                last_update = current_time

            should_checkpoint = (
                processed % checkpoint_interval == 0 and processed > 0
            ) or (current_time - last_checkpoint >= checkpoint_time_interval)

            if should_checkpoint and processed < total and index_dir:
                try:
                    log_structured("info", "checkpoint_saving", processed=processed)
                    yield f"data: {json.dumps({'status': 'checkpoint', 'message': 'Saving checkpoint...', 'processed': processed})}\n\n"

                    store = rebuild_service.index.store
                    if hasattr(store, "save"):
                        model_name = None
                        if hasattr(rebuild_service.index, "embedder") and hasattr(
                            rebuild_service.index.embedder, "model_name"
                        ):
                            model_name = rebuild_service.index.embedder.model_name

                        await asyncio.to_thread(
                            store.save, index_dir, model_name=model_name
                        )

                    if hasattr(rebuild_service.index, "save_registry"):
                        registry_path = Path(index_dir) / "doc_registry.json"
                        await asyncio.to_thread(
                            rebuild_service.index.save_registry, registry_path
                        )

                    log_structured(
                        "info", "checkpoint_saved", processed=processed, chunks=count
                    )
                    last_checkpoint = current_time

                except Exception as e:
                    log_structured("warning", "checkpoint_failed", error=str(e))

            if time.time() - last_update > 15:
                yield ": keep-alive\n\n"
                last_update = time.time()

        log_structured("info", "index_saving")
        yield f"data: {json.dumps({'status': 'saving', 'message': 'Saving index to disk...'})}\n\n"

        try:
            store = rebuild_service.index.store
            if hasattr(store, "save") and index_dir:
                model_name = None
                if hasattr(rebuild_service.index, "embedder") and hasattr(
                    rebuild_service.index.embedder, "model_name"
                ):
                    model_name = rebuild_service.index.embedder.model_name

                await asyncio.to_thread(store.save, index_dir, model_name=model_name)
                log_structured("info", "index_saved", path=index_dir)

                if hasattr(rebuild_service.index, "save_registry"):
                    registry_path = Path(index_dir) / "doc_registry.json"
                    await asyncio.to_thread(
                        rebuild_service.index.save_registry, registry_path
                    )
                    log_structured("info", "registry_saved", path=str(registry_path))
        except Exception as e:
            log_structured("error", "index_save_failed", error=str(e))
            yield f"data: {json.dumps({'status': 'save_error', 'error': str(e)})}\n\n"

        total_time = time.time() - start_time
        log_structured(
            "info",
            "index_rebuild_completed",
            chunks=count,
            files=processed,
            errors=errors,
            skipped=skipped,
            needs_ocr=needs_ocr_count,
            duration_s=round(total_time, 2),
        )

        final_data = {
            "status": "complete",
            "indexed": count,
            "total": processed,
            "errors": errors,
            "skipped": skipped,
            "needs_ocr": needs_ocr_count,
            "ocr_files": ocr_files[:20] if needs_ocr_count > 0 else [],
            "total_time_seconds": round(total_time, 2),
            "message": f"Indexed {count} chunks from {processed} files ({skipped} skipped, {errors} errors, {needs_ocr_count} need OCR)",
        }
        yield f"data: {json.dumps(final_data)}\n\n"

    except asyncio.CancelledError:
        log_structured("info", "index_rebuild_cancelled")
        yield f"data: {json.dumps({'status': 'cancelled', 'message': 'Rebuild cancelled'})}\n\n"
        raise
    except Exception as e:
        log_structured("error", "index_rebuild_failed", error=str(e))
        yield f"data: {json.dumps({'status': 'failed', 'error': str(e)})}\n\n"
