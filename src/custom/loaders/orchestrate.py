import logging
from src.custom.chunker.arxivchunker import TextChunker
from src.custom.embeddings.jarxivembeddings import JinaEmbeddingsService
from src.custom.loaders.opensearch import OpenSearchService
from src.custom.loaders.hybridsearchindex import HybridIndexingService

logger = logging.getLogger(__name__)


def build_hybrid_indexing_service() -> HybridIndexingService:
    """
    Factory function to build a production-ready HybridIndexingService.

    This function wires together:
    - TextChunker
    - Jina Embeddings Client
    - OpenSearch Client
    - HybridIndexingConfig

    :return: Fully configured HybridIndexingService instance
    """
    logger.info("Building HybridIndexingService via factory")

    chunker = TextChunker(
        chunk_size=500,
        overlap_size=100,
        min_chunk_size=120,
    )

    embeddings_client = JinaEmbeddingsService()
    opensearch_client = OpenSearchService()

    config = HybridIndexingConfig(
        embedding_batch_size=50,
        embedding_model="jina-embeddings-v3",
    )

    logger.info(
        "HybridIndexingService configuration created",
        extra={
            "chunk_size": 500,
            "overlap_size": 100,
            "min_chunk_size": 120,
            "embedding_batch_size": 50,
            "embedding_model": "jina-embeddings-v3",
        },
    )

    service = HybridIndexingService(
        chunker=chunker,
        embeddings_client=embeddings_client,
        opensearch_client=opensearch_client,
        config=config,
    )

    logger.info("HybridIndexingService successfully built")

    return service
