from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence

try:
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover
    from pydantic.v1 import BaseSettings  # fallback, if needed

class AppSettings(BaseSettings):
    app_name: str = "AI Obsidian Service"

    # Index / storage
    index_dir: Optional[str] = None

    # Logging
    json_logs_enabled: bool = True
    log_level: str = "INFO"
    log_uvicorn: bool = True

    # Request ID
    request_id_header: str = "X-Request-ID"

    class Config:
        env_prefix = "AI_OBSIDIAN_"
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
