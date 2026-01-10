import logging
from typing import Any, Dict, List, Optional

from opensearchpy import OpenSearch
from src.custom.loaders.helpers.indexconfig import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE
from src.custom.loaders.helpers.querybuilder import QueryBuilder
from src.custom.connectors.opensearch import OpenSearchConnector

logger = logging.getLogger(__name__)


class OpenSearchService:
    """
    OpenSearch Service Layer.

    Responsibilities
    ----------------
    - BM25 search
    - Vector search
    - Hybrid (BM25 + vector) search with RRF
    - Index creation & management
    - Chunk indexing & deletion

    """

    def __init__(self, connector: OpenSearchConnector, config: Dict[str, Any]):
        """
        Initialize OpenSearch service.

        Parameters
        ----------
        connector : OpenSearchConnector
            Initialized OpenSearch connector
        config : dict
            Required keys:
            - index_name
            - chunk_index_suffix
            - rrf_pipeline_id
        """
        self.client: OpenSearch = connector.connect()

        self.index_name = f"{config['index_name']}-{config['chunk_index_suffix']}"
        self.rrf_pipeline_id = config["rrf_pipeline_id"]

        logger.info("OpenSearchService initialized | index=%s", self.index_name)

   
    def health_check(self) -> bool:
        """
        Check OpenSearch cluster health.

        Returns:
            True if cluster status is green or yellow, False otherwise.
        """
        try:
            health = self.client.cluster.health()
            return health["status"] in ["green", "yellow"]
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return False

   
    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """
        Create the hybrid index and RRF search pipeline.

        Args:
            force: If True, existing index and pipeline will be recreated.

        Returns:
            Dictionary indicating creation status of index and pipeline.
        """
        return {
            "hybrid_index": self._create_hybrid_index(force),
            "rrf_pipeline": self._create_rrf_pipeline(force),
        }

    def _create_hybrid_index(self, force: bool = False) -> bool:
        """
        Create the hybrid OpenSearch index.

        Args:
            force: If True, delete and recreate the index.

        Returns:
            True if index was created, False if it already existed.
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info("Deleted index: %s", self.index_name)

            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(
                    index=self.index_name,
                    body=ARXIV_PAPERS_CHUNKS_MAPPING,
                )
                logger.info("Created hybrid index: %s", self.index_name)
                return True

            logger.info("Index already exists: %s", self.index_name)
            return False

        except Exception as e:
            logger.error("Index creation failed: %s", e)
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """
        Create the Reciprocal Rank Fusion (RRF) search pipeline.

        Args:
            force: If True, delete and recreate the pipeline.

        Returns:
            True if pipeline was created, False if it already existed.
        """
        pipeline_id = self.rrf_pipeline_id

        try:
            if force:
                try:
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info("Deleted pipeline: %s", pipeline_id)
                except Exception:
                    pass

            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info("RRF pipeline already exists: %s", pipeline_id)
                return False
            except Exception:
                pass

            self.client.transport.perform_request(
                "PUT",
                f"/_search/pipeline/{pipeline_id}",
                body={
                    "description": HYBRID_RRF_PIPELINE["description"],
                    "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"],
                },
            )

            logger.info("Created RRF pipeline: %s", pipeline_id)
            return True

        except Exception as e:
            logger.error("RRF pipeline creation failed: %s", e)
            raise

    # Search APIs

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = True,
        ) -> Dict[str, Any]:
        """
        Perform BM25-only keyword search.

        Args:
            query: Search text.
            size: Number of results to return.
            from_: Pagination offset.
            categories: Optional category filter.
            latest: Sort by published date if True.

        Returns:
            Search results dictionary.
        """
        return self._search_bm25_only(query, size, from_, categories, latest)

    def search_chunks_vector(
        self,
        query_embedding: List[float],
        size: int = 10,
        categories: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
        """
        Perform pure vector (semantic) search on chunks.

        Args:
            query_embedding: Embedding vector for the query.
            size: Number of results to return.
            categories: Optional category filter.

        Returns:
            Search results dictionary.
        """
        try:
            query = {"knn": {"embedding": {"vector": query_embedding, "k": size}}}

            if categories:
                query = {
                    "bool": {
                        "must": [query],
                        "filter": [{"terms": {"categories": categories}}],
                    }
                }

            response = self.client.search(
                index=self.index_name,
                body={
                    "size": size,
                    "query": query,
                    "_source": {"excludes": ["embedding"]},
                },
            )

            return self._parse_hits(response)

        except Exception as e:
            logger.error("Vector search failed: %s", e)
            return {"total": 0, "hits": []}

    def search_unified(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        size: int = 10,
        from_: int = 0,
        categories: Optional[List[str]] = None,
        latest: bool = False,
        use_hybrid: bool = True,
        min_score: float = 0.0,
        ) -> Dict[str, Any]:
        """
        Perform unified search (BM25, vector, or hybrid).

        Args:
            query: Search text.
            query_embedding: Optional query embedding.
            size: Number of results to return.
            from_: Pagination offset.
            categories: Optional category filter.
            latest: Sort by date instead of relevance.
            use_hybrid: Enable hybrid search if embedding is provided.
            min_score: Minimum score threshold.

        Returns:
            Search results dictionary.
        """
        if not query_embedding or not use_hybrid:
            return self._search_bm25_only(query, size, from_, categories, latest)

        return self._search_hybrid_native(
            query, query_embedding, size, categories, min_score
        )

    # Internal search helpers
 
    def _search_bm25_only(
        self,
        query: str,
        size: int,
        from_: int,
        categories: Optional[List[str]],
        latest: bool,
        ) -> Dict[str, Any]:
        """
        Execute BM25 keyword search using QueryBuilder.
        """
        builder = QueryBuilder(
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest_papers=latest,
            search_chunks=True,
        )

        response = self.client.search(
            index=self.index_name,
            body=builder.build(),
        )

        return self._parse_hits(response)

    def _search_hybrid_native(
        self,
        query: str,
        query_embedding: List[float],
        size: int,
        categories: Optional[List[str]],
        min_score: float,
        ) -> Dict[str, Any]:

        """
        Execute native hybrid search using BM25 + vector queries
        combined via RRF pipeline.
        """
        builder = QueryBuilder(
            query=query,
            size=size * 2,
            from_=0,
            categories=categories,
            latest_papers=False,
            search_chunks=True,
        )

        bm25_query = builder.build()["query"]

        hybrid_query = {
            "hybrid": {
                "queries": [
                    bm25_query,
                    {"knn": {"embedding": {"vector": query_embedding, "k": size * 2}}},
                ]
            }
        }

        response = self.client.search(
            index=self.index_name,
            body={"size": size, "query": hybrid_query},
            params={"search_pipeline": self.rrf_pipeline_id},
        )

        hits = []
        for hit in response["hits"]["hits"]:
            if hit["_score"] >= min_score:
                doc = hit["_source"]
                doc["score"] = hit["_score"]
                doc["chunk_id"] = hit["_id"]
                hits.append(doc)

        return {"total": len(hits), "hits": hits}

    # Indexing APIs
    #add client
  
    def index_chunk(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        
        """
        Index a single chunk document.

        Args:
            chunk_data: Chunk metadata and text.
            embedding: Embedding vector.

        Returns:
            True if indexing succeeded, False otherwise.
        """
        try:

        
            chunk_data["embedding"] = embedding
            res = self.client.index(index=self.index_name, body=chunk_data, refresh=True)
            return res["result"] in ["created", "updated"]
        except Exception as e:
            logger.error("Index chunk failed: %s", e)
            return False

    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        
        """
        Bulk index multiple chunks.

        Each item must contain:
            - chunk_data (dict)
            - embedding (list[float])

        Returns:
            Dictionary with success and failed counts.
        """
        from opensearchpy import helpers

        actions = []
        for item in chunks:
            data = item["chunk_data"].copy()
            data["embedding"] = item["embedding"]
            actions.append({"_index": self.index_name, "_source": data})

        success, failed = helpers.bulk(self.client, actions, refresh=True,  raise_on_error=False,)
        logger.info("Bulk indexed %d chunks, %d failed", success, len(failed))
        return {"success": success, "failed": len(failed)}

    def delete_paper_chunks(self, arxiv_id: str) -> bool:

        """
        Delete all chunks belonging to a paper.

        Args:
            arxiv_id: ArXiv paper identifier.

        Returns:
            True if any documents were deleted.
        """
        try:
            res = self.client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"arxiv_id": arxiv_id}}},
                refresh=True,
            )
            return res.get("deleted", 0) > 0
        except Exception as e:
            logger.error("Delete chunks failed: %s", e)
            return False

    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        
        """
        Retrieve all chunks for a given paper.

        Args:
            arxiv_id: ArXiv paper identifier.

        Returns:
            List of chunk documents sorted by chunk_index.
        """
        try:
            response = self.client.search(
                index=self.index_name,
                body={
                    "query": {"term": {"arxiv_id": arxiv_id}},
                    "size": 1000,
                    "sort": [{"chunk_index": "asc"}],
                    "_source": {"excludes": ["embedding"]},
                },
            )
            return [
                {**hit["_source"], "chunk_id": hit["_id"]}
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            logger.error("Fetch chunks failed: %s", e)
            return []

    # Utils

    def _parse_hits(self, response: Dict[str, Any]) -> Dict[str, Any]:

        """
        Parse OpenSearch search response into a clean format.
        """
        hits = []
        for hit in response["hits"]["hits"]:
            doc = hit["_source"]
            doc["score"] = hit["_score"]
            doc["chunk_id"] = hit["_id"]
            hits.append(doc)

        return {
            "total": response["hits"]["total"]["value"],
            "hits": hits,
        }
