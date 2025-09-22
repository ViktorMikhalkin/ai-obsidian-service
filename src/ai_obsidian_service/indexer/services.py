from ai_obsidian_service.adapters.services.search_service import SearchService
from ai_obsidian_service.config.container import build_search_service

__all__ = ["build_search_service", "SearchService"]


class IndexerService(SearchService):
    pass


__all__ = ["build_search_service", "SearchService", "IndexerService"]
