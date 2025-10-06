"""Configuration management endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ai_obsidian_service.api.dependencies import log_structured

router = APIRouter()

# Default config file location
CONFIG_FILE = Path("config/service_config.json")


class ChunkingConfig(BaseModel):
    """Chunking strategy configuration."""

    use_token_chunking: bool = Field(
        False,
        description="Use token-based semantic chunking (True) or character-based simple chunking (False)",
    )
    # Token-based settings (for SemanticChunker)
    target_tokens: int = Field(
        300, ge=50, le=2000, description="Target tokens per chunk (token-based)"
    )
    overlap_tokens: int = Field(
        60, ge=0, le=500, description="Token overlap between chunks (token-based)"
    )

    # Character-based settings (for SimpleChunker)
    max_chars: int = Field(
        1500, ge=100, le=10000, description="Maximum characters per chunk (char-based)"
    )
    overlap_chars: int = Field(
        150, ge=0, le=1000, description="Character overlap between chunks (char-based)"
    )

    @field_validator("overlap_tokens")
    @classmethod
    def validate_token_overlap(cls, v: int, info) -> int:
        if "target_tokens" in info.data and v >= info.data["target_tokens"]:
            raise ValueError("overlap_tokens must be less than target_tokens")
        return v

    @field_validator("overlap_chars")
    @classmethod
    def validate_char_overlap(cls, v: int, info) -> int:
        if "max_chars" in info.data and v >= info.data["max_chars"]:
            raise ValueError("overlap_chars must be less than max_chars")
        return v


class ParsingConfig(BaseModel):
    """Document parsing and file selection configuration."""

    # File type support
    enable_pdf: bool = Field(True, description="Enable PDF parsing")
    enable_markdown: bool = Field(True, description="Enable Markdown parsing")
    enable_epub: bool = Field(True, description="Enable EPUB parsing")

    # File filtering
    include_globs: list[str] = Field(
        default_factory=lambda: ["**/*.md", "**/*.pdf", "**/*.epub"],
        description="File patterns to include",
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            "**/.trash/**",
            "**/.obsidian/**",
            "**/archive/**",
            "**/.git/**",
            "**/node_modules/**",
            "**/*.png",
            "**/*.jpg",
            "**/*.jpeg",
            "**/*.gif",
        ],
        description="File patterns to exclude",
    )


class EmbeddingsConfig(BaseModel):
    """Embedding model and hardware configuration."""

    model: str | None = Field(None, description="Sentence transformer model name")
    device: str = Field("cpu", description="Device for embeddings: 'cpu' or 'cuda'")
    dtype: str = Field("float32", description="Data type: 'float32' or 'fp32'")
    batch_size: int = Field(
        32, ge=1, le=512, description="Batch size for embedding generation"
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        if v not in ("float32", "fp32"):
            raise ValueError("dtype must be 'float32' or 'fp32'")
        return v


class FAISSConfig(BaseModel):
    """FAISS-specific configuration."""

    search_on: str = Field(
        "cpu", description="Runtime FAISS search device: 'cpu' or 'gpu'"
    )
    index_path: str = Field("index/faiss.index", description="Path to FAISS index file")
    dim_path: str = Field(
        "index/dim.txt", description="Path to dimension metadata file"
    )

    @field_validator("search_on")
    @classmethod
    def validate_search_device(cls, v: str) -> str:
        if v not in ("cpu", "gpu"):
            raise ValueError("search_on must be 'cpu' or 'gpu'")
        return v


class IndexingConfig(BaseModel):
    """Vector indexing and storage configuration."""

    # Vector store backend
    backend: str | None = Field(
        None, description="Vector store backend: 'memory' or 'faiss'"
    )

    # Index persistence
    index_dir: str | None = Field(
        None, description="Directory for persisting the index"
    )

    # Incremental indexing
    enable_incremental: bool = Field(
        True, description="Enable incremental indexing (skip unchanged files)"
    )
    checkpoint_interval: int = Field(
        50, ge=1, description="Save checkpoint every N files during rebuild"
    )

    # FAISS-specific settings
    # FIX: Use Field with default instead of default_factory for nested models
    faiss: FAISSConfig = Field(default=FAISSConfig())

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str | None) -> str | None:
        if v is not None and v not in ("memory", "faiss"):
            raise ValueError("backend must be 'memory' or 'faiss'")
        return v


class BM25Config(BaseModel):
    """BM25 reranking configuration."""

    enabled: bool = Field(True, description="Enable BM25 reranking")
    top_n: int = Field(50, ge=1, le=200, description="Number of candidates to rerank")


class VaultConfig(BaseModel):
    """Vault and library paths configuration."""

    vault_path: str | None = Field(None, description="Primary Obsidian vault path")
    library_paths: list[str] = Field(
        default_factory=list, description="Additional library paths to index"
    )


class ServerConfig(BaseModel):
    """API server configuration."""

    host: str = Field("127.0.0.1", description="Server host address")
    port: int = Field(8000, ge=1, le=65535, description="Server port")


class ServiceConfig(BaseModel):
    """Complete service configuration."""

    # Core configurations
    # FIX: Use Field with default instead of default_factory for nested Pydantic models
    vault: VaultConfig = Field(default=VaultConfig())
    parsing: ParsingConfig = Field(default=ParsingConfig())
    chunking: ChunkingConfig = Field(default=ChunkingConfig())
    embeddings: EmbeddingsConfig = Field(default=EmbeddingsConfig())
    indexing: IndexingConfig = Field(default=IndexingConfig())
    bm25: BM25Config = Field(default=BM25Config())
    server: ServerConfig = Field(default=ServerConfig())

    # LLM settings for RAG
    ollama_base_url: str | None = Field(None, description="Ollama API base URL")
    ollama_model: str | None = Field(None, description="Ollama model name")
    ollama_timeout: float = Field(
        30.0, ge=1.0, le=300.0, description="Ollama request timeout in seconds"
    )

    # Logging
    log_level: str = Field("INFO", description="Logging level")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper


# Global config state
_current_config: ServiceConfig | None = None


def get_current_config() -> ServiceConfig:
    """
    Get current active configuration, loading from disk if needed.
    Returns default config with warnings if file doesn't exist.
    """
    global _current_config

    if _current_config is None:
        if not CONFIG_FILE.exists():
            # FIX: Remove 'message' keyword to avoid conflict
            log_structured(
                "warning",
                "config_not_found",
                path=str(CONFIG_FILE),
                details="Using default configuration. Set config via POST /config",
            )
            _current_config = ServiceConfig()
        else:
            try:
                with open(CONFIG_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                _current_config = ServiceConfig(**data)
                log_structured("info", "config_loaded_from_disk", path=str(CONFIG_FILE))
            except Exception as e:
                # FIX: Remove 'message' keyword to avoid conflict
                log_structured(
                    "error",
                    "config_load_failed",
                    path=str(CONFIG_FILE),
                    error=str(e),
                    details="Using default configuration",
                )
                _current_config = ServiceConfig()

    return _current_config


def require_config_field(field_path: str, value: Any, operation: str) -> None:
    """
    Validate that a required config field is set for a specific operation.
    Raises HTTPException if the field is None or empty.

    Args:
        field_path: Dot-notation path to field (e.g., "indexing.index_dir")
        value: The field value to check
        operation: Operation name for error message (e.g., "index rebuild")
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "CONFIG_REQUIRED",
                "message": f"Configuration field '{field_path}' must be set to perform {operation}",
                "field": field_path,
                "operation": operation,
                "help": f"Set configuration via POST /config with required '{field_path}' field",
            },
        )


def save_config(config: ServiceConfig) -> None:
    """Persist configuration to disk."""
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2, ensure_ascii=False)
        log_structured("info", "config_saved_to_disk", path=str(CONFIG_FILE))
    except Exception as e:
        log_structured(
            "error", "config_save_failed", path=str(CONFIG_FILE), error=str(e)
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CONFIG_SAVE_FAILED",
                "message": f"Failed to save configuration: {str(e)}",
            },
        ) from e


@router.get("/config", response_model=ServiceConfig)
def get_config() -> ServiceConfig:
    """
    Get current active configuration.

    Returns the configuration that is currently in use by the service.
    If no config file exists, returns default configuration with warnings logged.
    """
    try:
        config = get_current_config()

        # Generate warnings for missing critical fields
        warnings = []
        if not config.vault.vault_path:
            warnings.append(
                "vault.vault_path not set (required for indexing operations)"
            )
        if not config.indexing.backend:
            warnings.append(
                "indexing.backend not set (required for indexing operations)"
            )
        if not config.embeddings.model:
            warnings.append(
                "embeddings.model not set (required for indexing operations)"
            )
        if not config.indexing.index_dir:
            warnings.append(
                "indexing.index_dir not set (required for index persistence)"
            )

        if warnings:
            log_structured("warning", "config_incomplete", warnings=warnings)

        log_structured("info", "config_retrieved", warnings_count=len(warnings))
        return config

    except Exception as e:
        log_structured("error", "config_retrieval_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={"code": "CONFIG_RETRIEVAL_FAILED", "message": str(e)},
        ) from e


@router.post("/config", response_model=dict[str, Any])
def set_config(config: ServiceConfig):
    """
    Set new service configuration.

    - Validates all settings using Pydantic models
    - Persists to disk (config/service_config.json)
    - Settings apply to NEW operations only (ongoing operations continue with previous config)
    - Returns confirmation with warnings about when changes take effect

    Note: Some changes (like embedding model or backend) may require service restart
    and index rebuild to take full effect.
    """
    global _current_config

    try:
        # Validate by attempting to create the model (already done by FastAPI, but explicit)
        validated_config = ServiceConfig(**config.model_dump())

        # Save to disk
        save_config(validated_config)

        # Update in-memory config
        _current_config = validated_config

        log_structured(
            "info",
            "config_updated",
            backend=validated_config.indexing.backend,
            model=validated_config.embeddings.model,
            index_dir=validated_config.indexing.index_dir,
        )

        warnings = _generate_warnings(validated_config)
        setup_warnings = _generate_setup_warnings(validated_config)

        return {
            "status": "success",
            "message": "Configuration updated successfully",
            "note": "New settings will apply to future operations. Ongoing operations continue with previous settings.",
            "warnings": warnings,
            "setup_warnings": setup_warnings,
            "config": validated_config.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        log_structured("error", "config_update_failed", error=str(e))
        raise HTTPException(
            status_code=500, detail={"code": "CONFIG_UPDATE_FAILED", "message": str(e)}
        ) from e


def _generate_setup_warnings(config: ServiceConfig) -> list[str]:
    """Generate warnings about missing required configuration."""
    warnings = []

    if not config.vault.vault_path:
        warnings.append(
            "vault.vault_path not set - indexing operations require a vault path to index from"
        )
    if not config.indexing.backend:
        warnings.append(
            "indexing.backend not set - indexing operations will fail until configured"
        )
    if not config.embeddings.model:
        warnings.append(
            "embeddings.model not set - indexing operations will fail until configured"
        )
    if not config.indexing.index_dir:
        warnings.append(
            "indexing.index_dir not set - index persistence will fail until configured"
        )
    if not config.ollama_base_url or not config.ollama_model:
        warnings.append(
            "Ollama settings incomplete - RAG answer generation will be unavailable"
        )

    return warnings


def _generate_warnings(config: ServiceConfig) -> list[str]:
    """Generate warnings about configuration changes that may require additional actions."""
    warnings = []

    # Check if changing embedding model or other critical settings
    try:
        old_config = _current_config
        if old_config is None:
            return []

        if old_config.embeddings.model != config.embeddings.model:
            warnings.append(
                "Embedding model changed. Existing index may be incompatible. "
                "Consider running index rebuild with force=true."
            )
        if old_config.indexing.backend != config.indexing.backend:
            warnings.append(
                "Vector store backend changed. Service restart and index rebuild required."
            )
        if old_config.chunking.use_token_chunking != config.chunking.use_token_chunking:
            warnings.append(
                "Chunking strategy changed (token vs char-based). "
                "Full index rebuild required for consistency."
            )
        if (
            old_config.chunking.max_chars != config.chunking.max_chars
            or old_config.chunking.overlap_chars != config.chunking.overlap_chars
            or old_config.chunking.target_tokens != config.chunking.target_tokens
            or old_config.chunking.overlap_tokens != config.chunking.overlap_tokens
        ):
            warnings.append(
                "Chunking settings changed. New settings apply to newly indexed documents only. "
                "For consistency, consider full index rebuild."
            )
        if old_config.embeddings.device != config.embeddings.device:
            warnings.append(
                "Embedding device changed (CPU/CUDA). Service restart recommended."
            )
    except Exception:
        pass

    return warnings
