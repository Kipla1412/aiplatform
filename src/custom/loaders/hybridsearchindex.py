import logging
from typing import Dict, List, Any, Optional
from src.custom.loaders.schemas.hybridsearch import HybridIndexingConfig

logger = logging.getLogger(__name__)


class HybridIndexingService:
    """
    Reusable Hybrid Indexing Service.

    Responsibilities:
    - Chunk research paper text
    - Generate vector embeddings
    - Prepare OpenSearch payload
    - Bulk index chunks into OpenSearch
    - Support batch indexing and re-indexing
    """

    def __init__(
        self,
        chunker,
        embeddings_client,
        opensearch_client,
        config: Optional[HybridIndexingConfig] = None,
    ):
        """
        Initialize HybridIndexingService.

        :param chunker: Text chunking service
        :param embeddings_client: Client for generating embeddings
        :param opensearch_client: OpenSearch client for indexing
        :param config: Optional hybrid indexing configuration
        """
        self.chunker = chunker
        self.embeddings_client = embeddings_client
        self.opensearch_client = opensearch_client
        self.config = config or HybridIndexingConfig()

        logger.info(
            "HybridIndexingService initialized",
            extra={
                "embedding_model": self.config.embedding_model,
                "batch_size": self.config.embedding_batch_size,
            },
        )

    # ------------------------------
    # PUBLIC APIs
    # ------------------------------

    async def index_paper(self, paper_data: Dict[str, Any]) -> Dict[str, int]:
        """
        Index a single paper using a hybrid search pipeline.

        Steps:
        1) Chunk the paper
        2) Generate embeddings
        3) Prepare OpenSearch payload
        4) Bulk index into OpenSearch

        :param paper_data: Dictionary containing paper metadata and text
        :return: Dictionary with indexing statistics
        """
        arxiv_id = paper_data.get("arxiv_id")
        paper_id = str(paper_data.get("id", ""))

        if not arxiv_id:
            logger.error(
                "Cannot index paper: missing arxiv_id",
                extra={"paper_id": paper_id},
            )
            return {
                "chunks_created": 0,
                "chunks_indexed": 0,
                "embeddings_generated": 0,
                "errors": 1,
            }

        logger.info(
            "Starting indexing pipeline",
            extra={"arxiv_id": arxiv_id, "paper_id": paper_id},
        )

        try:
            # -------- STEP 1: Chunking --------
            logger.info(
                "Chunking paper",
                extra={"arxiv_id": arxiv_id},
            )

            chunks = self.chunker.chunk_paper(
                title=paper_data.get("title", ""),
                abstract=paper_data.get("abstract", ""),
                full_text=paper_data.get(
                    "raw_text", paper_data.get("full_text", "")
                ),
                arxiv_id=arxiv_id,
                paper_id=paper_id,
                sections=paper_data.get("sections"),
            )

            if not chunks:
                logger.warning(
                    "No chunks created for paper",
                    extra={"arxiv_id": arxiv_id},
                )
                return {
                    "chunks_created": 0,
                    "chunks_indexed": 0,
                    "embeddings_generated": 0,
                    "errors": 0,
                }

            logger.info(
                "Chunking completed",
                extra={"arxiv_id": arxiv_id, "num_chunks": len(chunks)},
            )

            # -------- STEP 2: Generate Embeddings --------
            logger.info(
                "Generating embeddings",
                extra={
                    "arxiv_id": arxiv_id,
                    "num_chunks": len(chunks),
                    "model": self.config.embedding_model,
                },
            )

            chunk_texts = [chunk.text for chunk in chunks]

            embeddings = await self.embeddings_client.embed_passages(
                texts=chunk_texts,
                batch_size=self.config.embedding_batch_size,
            )

            if len(embeddings) != len(chunks):
                logger.error(
                    "Embedding count mismatch",
                    extra={
                        "arxiv_id": arxiv_id,
                        "num_chunks": len(chunks),
                        "num_embeddings": len(embeddings),
                    },
                )
                return {
                    "chunks_created": len(chunks),
                    "chunks_indexed": 0,
                    "embeddings_generated": len(embeddings),
                    "errors": 1,
                }

            logger.info(
                "Embeddings generated successfully",
                extra={
                    "arxiv_id": arxiv_id,
                    "num_embeddings": len(embeddings),
                },
            )

            # -------- STEP 3: Prepare payload for OpenSearch --------
            logger.debug(
                "Preparing OpenSearch payload",
                extra={"arxiv_id": arxiv_id},
            )

            chunks_with_embeddings = []

            for chunk, embedding in zip(chunks, embeddings):
                chunk_data = {
                    "arxiv_id": chunk.arxiv_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "section_title": chunk.metadata.section_title,
                    "embedding_model": self.config.embedding_model,
                    "title": paper_data.get("title", ""),
                    "authors": (
                        ", ".join(paper_data.get("authors", []))
                        if isinstance(paper_data.get("authors"), list)
                        else paper_data.get("authors", "")
                    ),
                    "abstract": paper_data.get("abstract", ""),
                    "categories": paper_data.get("categories", []),
                    "published_date": paper_data.get("published_date"),
                }

                chunks_with_embeddings.append(
                    {
                        "chunk_data": chunk_data,
                        "embedding": embedding,
                    }
                )

            logger.info(
                "Payload preparation completed",
                extra={
                    "arxiv_id": arxiv_id,
                    "payload_size": len(chunks_with_embeddings),
                },
            )

            # -------- STEP 4: Index to OpenSearch --------
            logger.info(
                "Starting bulk indexing",
                extra={"arxiv_id": arxiv_id},
            )

            results = self.opensearch_client.bulk_index_chunks(
                chunks_with_embeddings
            )

            logger.info(
                "Bulk indexing finished",
                extra={
                    "arxiv_id": arxiv_id,
                    "success": results["success"],
                    "failed": results["failed"],
                },
            )

            return {
                "chunks_created": len(chunks),
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }

        except Exception as e:
            logger.exception(
                "Unhandled error during indexing",
                extra={"arxiv_id": arxiv_id, "error": str(e)},
            )
            return {
                "chunks_created": 0,
                "chunks_indexed": 0,
                "embeddings_generated": 0,
                "errors": 1,
            }

    async def index_papers_batch(
        self, papers: List[Dict[str, Any]], replace_existing: bool = False
    ) -> Dict[str, int]:
        """
        Index multiple papers in a batch.

        :param papers: List of paper dictionaries
        :param replace_existing: If True, delete existing chunks before re-indexing
        :return: Aggregated batch statistics
        """
        logger.info(
            "Starting batch indexing",
            extra={
                "num_papers": len(papers),
                "replace_existing": replace_existing,
            },
        )

        total_stats = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for idx, paper in enumerate(papers, start=1):
            arxiv_id = paper.get("arxiv_id")

            logger.info(
                "Processing paper in batch",
                extra={"batch_index": idx, "arxiv_id": arxiv_id},
            )

            if replace_existing and arxiv_id:
                deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
                logger.info(
                    "Deleted existing chunks before reindexing",
                    extra={"arxiv_id": arxiv_id, "deleted": deleted},
                )

            stats = await self.index_paper(paper)

            total_stats["papers_processed"] += 1
            total_stats["total_chunks_created"] += stats["chunks_created"]
            total_stats["total_chunks_indexed"] += stats["chunks_indexed"]
            total_stats["total_embeddings_generated"] += stats[
                "embeddings_generated"
            ]
            total_stats["total_errors"] += stats["errors"]

        logger.info(
            "Batch indexing complete",
            extra={
                "papers_processed": total_stats["papers_processed"],
                "chunks_indexed": total_stats["total_chunks_indexed"],
                "errors": total_stats["total_errors"],
            },
        )

        return total_stats

    async def reindex_paper(
        self, arxiv_id: str, paper_data: Dict[str, Any]
    ) -> Dict[str, int]:
        """
        Reindex a paper:
        1) Delete old chunks
        2) Re-run indexing pipeline

        :param arxiv_id: Arxiv identifier
        :param paper_data: Updated paper content
        :return: Indexing statistics
        """
        logger.info(
            "Starting re-indexing",
            extra={"arxiv_id": arxiv_id},
        )

        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)

        if deleted:
            logger.info(
                "Deleted existing chunks for reindexing",
                extra={"arxiv_id": arxiv_id},
            )
        else:
            logger.warning(
                "No existing chunks found to delete",
                extra={"arxiv_id": arxiv_id},
            )

        return await self.index_paper(paper_data)
