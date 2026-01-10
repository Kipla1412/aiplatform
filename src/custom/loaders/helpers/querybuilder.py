import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryBuilder:
    """
    Builds BM25-based OpenSearch queries for paper and chunk search.

    This class is responsible for constructing OpenSearch Query DSL
    including text search, filters, sorting, pagination, and highlighting.
    """

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
        search_chunks: bool = False,
    ):
        """
        Initialize query builder.

        Args:
            query: Search text entered by the user.
            size: Number of results to return.
            from_: Offset for pagination.
            fields: Fields to search (auto-selected if None).
            categories: Optional category filter.
            track_total_hits: Track accurate total hit count.
            latest_papers: Sort by published date instead of relevance.
            search_chunks: Enable chunk-level search behavior.
        """
        self.query = query
        self.size = size
        self.from_ = from_
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers
        self.search_chunks = search_chunks

        # Auto-select fields with boosts
        if fields is None:
            self.fields = (
                ["chunk_text^3", "title^2", "abstract^1"]
                if search_chunks
                else ["title^3", "abstract^2", "authors^1"]
            )
        else:
            self.fields = fields

    def build(self) -> Dict[str, Any]:
        """
        Build the full OpenSearch query body.

        Returns
        -------
        dict
            Complete OpenSearch Query DSL payload including:
            - query
            - pagination
            - source filtering
            - highlighting
            - sorting (if applicable)
        """
        body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        sort = self._build_sort()
        if sort:
            body["sort"] = sort

        return body

    def _build_query(self) -> Dict[str, Any]:
        """
        Construct the main boolean query structure.

        Returns
        -------
        dict
            Bool query containing:
            - must clauses for BM25 text search
            - filter clauses for category filtering
        """
        must = []
        if self.query.strip():
            must.append(self._build_text_query())

        query: Dict[str, Any] = {
            "must": must if must else [{"match_all": {}}]
        }

        filters = self._build_filters()
        if filters:
            query["filter"] = filters

        return {"bool": query}

    def _build_text_query(self) -> Dict[str, Any]:
        """
        Build the BM25 multi-field text query.

        Returns
        -------
        dict
            OpenSearch `multi_match` query with:
            - field boosting
            - fuzziness for typo tolerance
            - best_fields scoring strategy
        """
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """
        Build non-scoring filter clauses.

        Returns
        -------
        list[dict]
            List of filter queries applied in filter context
            (cached, fast, non-scoring).
        """
        return (
            [{"terms": {"categories": self.categories}}]
            if self.categories
            else []
        )

    def _build_source_fields(self) -> Any:
        """
        Define which document fields are returned in the response.

        Returns
        
        dict or list
            - For chunk search: excludes embedding vectors
            - For paper search: returns selected metadata fields
        """
        if self.search_chunks:
            return {"excludes": ["embedding"]}

        return [
            "arxiv_id",
            "title",
            "authors",
            "abstract",
            "categories",
            "published_date",
            "pdf_url",
        ]

    def _build_highlight(self) -> Dict[str, Any]:
        """
        Configure result highlighting.

        Returns
        -------
        dict
            Highlight configuration optimized for either:
            - Chunk-level previews
            - Paper-level previews
        """
        if self.search_chunks:
            return {
                "fields": {
                    "chunk_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "title": {"fragment_size": 0, "number_of_fragments": 0},
                    "abstract": {
                        "fragment_size": 150,
                        "number_of_fragments": 1,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            }

        return {
            "fields": {
                "title": {"fragment_size": 0, "number_of_fragments": 0},
                "abstract": {
                    "fragment_size": 150,
                    "number_of_fragments": 3,
                    "pre_tags": ["<mark>"],
                    "post_tags": ["</mark>"],
                },
                "authors": {"fragment_size": 0, "number_of_fragments": 0},
            },
            "require_field_match": False,
        }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """
        Build sorting configuration.

        Returns
     
        list[dict] or None
            Sorting rules if sorting is required,
            otherwise None to allow default relevance sorting.
        """
        if self.latest_papers:
            return [{"published_date": {"order": "desc"}}, "_score"]

        return None if self.query.strip() else [{"published_date": {"order": "desc"}}, "_score"]
