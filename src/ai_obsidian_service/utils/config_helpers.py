"""Helper utilities for working with configuration in endpoints."""

from typing import Any

from fastapi import HTTPException

from ai_obsidian_service.api.endpoints.config import (
    ServiceConfig,
    get_current_config,
    require_config_field,
)


class ConfigValidator:
    """
    Helper class for validating configuration requirements in endpoints.

    Usage in endpoints:
        validator = ConfigValidator()
        validator.require_indexing()  # Validates all indexing fields
        config = validator.config  # Access validated config
    """

    def __init__(self):
        self.config = get_current_config()

    def require_indexing(self) -> ServiceConfig:
        """
        Validate all required fields for indexing operations.

        Required fields:
        - vault.vault_path (where to index from)
        - indexing.backend (storage backend)
        - embeddings.model (embedding model)
        - indexing.index_dir (where to store index)

        Raises:
            HTTPException: If any required field is missing

        Returns:
            ServiceConfig: The validated configuration
        """
        require_config_field(
            "vault.vault_path", self.config.vault.vault_path, "indexing operations"
        )
        require_config_field(
            "indexing.backend", self.config.indexing.backend, "indexing operations"
        )
        require_config_field(
            "embeddings.model", self.config.embeddings.model, "indexing operations"
        )
        require_config_field(
            "indexing.index_dir", self.config.indexing.index_dir, "indexing operations"
        )
        return self.config

    def require_search(self) -> ServiceConfig:
        """
        Validate all required fields for search operations.

        Required fields:
        - indexing.backend
        - embeddings.model

        Raises:
            HTTPException: If any required field is missing

        Returns:
            ServiceConfig: The validated configuration
        """
        require_config_field(
            "indexing.backend", self.config.indexing.backend, "search operations"
        )
        require_config_field(
            "embeddings.model", self.config.embeddings.model, "search operations"
        )
        return self.config

    def require_rag(self) -> ServiceConfig:
        """
        Validate all required fields for RAG (answer) operations.

        Required fields:
        - indexing.backend
        - embeddings.model
        - ollama_base_url
        - ollama_model

        Raises:
            HTTPException: If any required field is missing

        Returns:
            ServiceConfig: The validated configuration
        """
        # First check search requirements
        self.require_search()

        # Then check LLM requirements
        require_config_field(
            "ollama_base_url", self.config.ollama_base_url, "RAG answer generation"
        )
        require_config_field(
            "ollama_model", self.config.ollama_model, "RAG answer generation"
        )
        return self.config

    def require_ocr(self) -> ServiceConfig:
        """
        Validate configuration for OCR operations.

        Required fields:
        - vault.vault_path (folder containing files to OCR)

        Note: OCR processes files in-place in the vault directory,
        adding searchable text layers to scanned PDFs.

        Raises:
            HTTPException: If vault_path is not set

        Returns:
            ServiceConfig: The validated configuration
        """
        require_config_field(
            "vault.vault_path", self.config.vault.vault_path, "OCR operations"
        )
        return self.config

    def require_field(self, field_path: str, operation: str) -> Any:
        """
        Validate a specific config field is set.

        Args:
            field_path: Dot-notation path (e.g., "indexing.index_dir")
            operation: Operation name for error message

        Returns:
            The field value

        Raises:
            HTTPException: If field is missing
        """
        parts = field_path.split(".")
        value = self.config

        for part in parts:
            if hasattr(value, part):
                value = getattr(value, part)
            else:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "code": "CONFIG_FIELD_NOT_FOUND",
                        "message": f"Config field '{field_path}' does not exist",
                        "field": field_path,
                    },
                )

        require_config_field(field_path, value, operation)
        return value


# Convenience function for simple cases
def validate_for_operation(operation: str) -> ServiceConfig:
    """
    Validate configuration based on operation type.

    Args:
        operation: One of 'indexing', 'search', 'rag', 'ocr'

    Returns:
        ServiceConfig: The validated configuration

    Raises:
        HTTPException: If required fields are missing
        ValueError: If operation type is unknown

    Usage:
        config = validate_for_operation('indexing')
        # Now safe to use config.indexing.index_dir etc.
    """
    validator = ConfigValidator()

    operation_map = {
        "indexing": validator.require_indexing,
        "search": validator.require_search,
        "rag": validator.require_rag,
        "answer": validator.require_rag,  # Alias
        "ocr": validator.require_ocr,
    }

    validate_func = operation_map.get(operation.lower())
    if not validate_func:
        raise ValueError(
            f"Unknown operation '{operation}'. "
            f"Valid operations: {', '.join(operation_map.keys())}"
        )

    return validate_func()
