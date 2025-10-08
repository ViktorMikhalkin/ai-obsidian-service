"""Index rebuild and duplicate detection endpoints."""

import json
from collections import defaultdict
from pathlib import Path
from typing import cast

from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse

from ai_obsidian_service.api.dependencies import _rebuild_lock
from ai_obsidian_service.api.endpoints.config import (
    get_current_config,
    require_config_field,
)
from ai_obsidian_service.api.logging import log_structured
from ai_obsidian_service.api.streaming import create_rebuild_stream
from ai_obsidian_service.utils.config_helpers import validate_for_operation

router = APIRouter()


@router.post("/rebuild")
async def index_rebuild(root: str = Body(..., embed=True), force: bool = Body(False)):
    """
    Rebuild index from a root directory with real-time progress updates via SSE.

    Args:
        root: Path to vault/library root directory
        force: If True, reindex all files ignoring registry (full rebuild)

    Supports:
    - Incremental indexing (skip unchanged files) when force=False
    - Full rebuild when force=True
    - Checkpoint saving (every 50 files)
    - Progress streaming
    - OCR detection tracking

    Requires configuration:
    - vault.vault_path must be set
    - indexing.backend must be set
    - embeddings.model must be set
    - indexing.index_dir must be set
    """

    # Validate required config FIRST
    config = validate_for_operation("indexing")

    if not root:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "MISSING_ROOT",
                "message": "Provide JSON body with 'root' field",
            },
        )

    p = Path(root)
    if not p.exists():
        log_structured(
            "warning", "index_rebuild_invalid_path", root=root, reason="not_found"
        )
        raise HTTPException(
            status_code=400,
            detail={"code": "ROOT_NOT_FOUND", "message": f"Path not found: {root}"},
        )
    if not p.is_dir():
        log_structured(
            "warning", "index_rebuild_invalid_path", root=root, reason="not_directory"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ROOT_NOT_DIR",
                "message": f"Path is not a directory: {root}",
            },
        )

    if _rebuild_lock.locked():
        log_structured("warning", "index_rebuild_rejected", reason="already_running")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "code": "INDEX_REBUILDING",
                "message": "Rebuild already in progress",
            },
        )

    # Use config.indexing.index_dir instead of os.getenv("INDEX_DIR")
    return StreamingResponse(
        create_rebuild_stream(root, config.indexing.index_dir, force),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/duplicates")
def find_duplicate_files(vault_root: str | None = None):
    """
    Find files with identical content based on doc_hash and file size.

    Uses both content hash (MD5) and file size to identify duplicates,
    reducing false positives from hash collisions.

    Args:
        vault_root: Optional vault root path to resolve relative paths

    Returns:
        Groups of files with identical content, potential space savings

    Requires configuration:
    - indexing.index_dir must be set
    """
    # Validate required config
    config = get_current_config()
    require_config_field(
        "indexing.index_dir", config.indexing.index_dir, "duplicate detection"
    )

    # Use config.indexing.index_dir
    index_dir = cast(str, config.indexing.index_dir)
    registry_path = Path(index_dir) / "doc_registry.json"

    if not registry_path.exists():
        return {
            "duplicate_groups": [],
            "total_duplicates": 0,
            "total_duplicate_files": 0,
            "potential_file_removals": 0,
            "total_size_savings_bytes": 0,
            "total_size_savings_mb": 0.0,
            "total_size_savings_gb": 0.0,
            "message": "Registry file not found. Run index rebuild first.",
        }

    try:
        with open(registry_path, encoding="utf-8") as f:
            registry = json.load(f)
    except Exception as e:
        log_structured(
            "error", "registry_load_failed", path=str(registry_path), error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "REGISTRY_LOAD_FAILED",
                "message": f"Failed to load registry: {str(e)}",
            },
        ) from e

    if len(registry) == 0:
        return {
            "duplicate_groups": [],
            "total_duplicates": 0,
            "total_duplicate_files": 0,
            "potential_file_removals": 0,
            "total_size_savings_bytes": 0,
            "total_size_savings_mb": 0.0,
            "total_size_savings_gb": 0.0,
            "message": "Registry is empty. Run index rebuild first.",
        }

    # Resolve vault root - use config if not provided
    if vault_root is None:
        vault_root = config.vault.vault_path
        if not vault_root:
            # Fallback to current directory if vault_path not set
            import os

            vault_root = os.getcwd()

    vault_path = Path(vault_root)

    # Group files by (hash, size) tuple for stronger duplicate detection
    hash_size_to_docs = defaultdict(list)

    for doc_id, doc_hash in registry.items():
        try:
            file_path = vault_path / doc_id
            if file_path.exists() and file_path.is_file():
                file_size = file_path.stat().st_size
                hash_size_to_docs[(doc_hash, file_size)].append(
                    {"path": doc_id, "size": file_size, "absolute_path": str(file_path)}
                )
            # Skip files that don't exist - don't include them in results
        except Exception as e:
            # Skip files with errors - don't pollute results with phantom files
            log_structured(
                "warning", "duplicate_check_file_error", doc_id=doc_id, error=str(e)
            )

    # Filter to only duplicates
    duplicates = {key: docs for key, docs in hash_size_to_docs.items() if len(docs) > 1}

    # Format response with detailed file info
    duplicate_groups = []
    total_size_saved = 0

    for (doc_hash, file_size), docs in sorted(
        duplicates.items(), key=lambda x: len(x[1]), reverse=True
    ):
        group = {
            "hash": doc_hash,
            "file_size": file_size if file_size != -1 else None,
            "count": len(docs),
            "files": docs,
        }

        if file_size > 0:
            savings = file_size * (len(docs) - 1)
            group["potential_savings_bytes"] = savings
            group["potential_savings_mb"] = round(savings / (1024 * 1024), 2)
            total_size_saved += savings

        duplicate_groups.append(group)

    log_structured(
        "info",
        "duplicates_found",
        groups=len(duplicate_groups),
        total_files=sum(g["count"] for g in duplicate_groups),
    )

    return {
        "duplicate_groups": duplicate_groups,
        "total_duplicates": len(duplicate_groups),
        "total_duplicate_files": sum(g["count"] for g in duplicate_groups),
        "potential_file_removals": sum(g["count"] - 1 for g in duplicate_groups),
        "total_size_savings_bytes": total_size_saved,
        "total_size_savings_mb": round(total_size_saved / (1024 * 1024), 2),
        "total_size_savings_gb": round(total_size_saved / (1024 * 1024 * 1024), 2),
        "vault_root": str(vault_path),
        "scanned_documents": len(registry),
    }
