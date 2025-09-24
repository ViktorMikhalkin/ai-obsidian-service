from __future__ import annotations
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # App
    app_name: str = "AI Obsidian Service"
    log_level: str = "INFO"
    json_logs_enabled: bool = True
    log_uvicorn: bool = True

    # Vector stack (DI v2)
    vector_store_backend: str = "memory"  # "memory" | "faiss"
    st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_dir: str | None = None   # e.g. "/var/data/obsidian_index"

    class Config:
        env_prefix = ""         # read vars as-is (no prefix)
        env_file = ".env"       # optional: load from .env if present
        case_sensitive = False  # nice to have for env names


_settings: Settings | None = None

def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
