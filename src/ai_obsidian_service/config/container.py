from ai_obsidian_service.adapters.chunkers.simple_chunker import SimpleChunker
from ai_obsidian_service.adapters.index.faiss_index import FaissIndex
from ai_obsidian_service.adapters.parsers import default_parsers
from ai_obsidian_service.adapters.services.search_service import SearchService


def build_search_service(index_dir: str | None = None) -> SearchService:
    return SearchService(
        parsers=default_parsers(),
        chunker=SimpleChunker(max_chars=1000, overlap=100),
        index=FaissIndex(index_dir=index_dir, dim=64),
    )
