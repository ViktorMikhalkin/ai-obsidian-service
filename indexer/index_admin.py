
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any
import os, time, json, yaml, subprocess, sys, datetime

CONFIG_PATH = Path("config.yaml")

def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

@dataclass
class IndexStats:
    index_path: str
    exists: bool
    size_bytes: Optional[int]
    last_modified: Optional[str]
    chunks_estimate: Optional[int]

def get_index_stats() -> IndexStats:
    cfg = _load_config()
    # default path for faiss index
    faiss_cfg = (cfg.get("faiss") or {})
    index_path = Path(faiss_cfg.get("index_path") or ".index/faiss.index")
    p = Path(index_path)
    exists = p.exists()
    size_bytes = p.stat().st_size if exists else None
    mtime = p.stat().st_mtime if exists else None
    last_modified = datetime.datetime.fromtimestamp(mtime).isoformat() if mtime else None

    # try to estimate chunks from a companion jsonl if present
    jsonl = Path("index/index.jsonl")
    chunks_estimate = None
    if jsonl.exists():
        try:
            with jsonl.open("r", encoding="utf-8", errors="ignore") as f:
                chunks_estimate = sum(1 for _ in f)
        except Exception:
            chunks_estimate = None

    return IndexStats(
        index_path=str(p),
        exists=exists,
        size_bytes=size_bytes,
        last_modified=last_modified,
        chunks_estimate=chunks_estimate,
    )

def rebuild_index(timeout_sec: int = 0) -> Dict[str, Any]:
    """Run CLI build to (re)create the index.
    If timeout_sec > 0, wait up to timeout for completion; otherwise fire-and-return.
    """
    cmd = [sys.executable, "-m", "cli.aiobs", "build"]
    start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    waited = False
    if timeout_sec and timeout_sec > 0:
        try:
            proc.wait(timeout=timeout_sec)
            waited = True
        except subprocess.TimeoutExpired:
            # still running in background
            pass
    return {
        "pid": proc.pid,
        "waited": waited,
        "timeout_sec": timeout_sec,
        "started_at": datetime.datetime.fromtimestamp(start).isoformat(),
        "cmd": " ".join(cmd),
    }
