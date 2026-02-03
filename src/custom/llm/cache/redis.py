import hashlib
import json
import logging
from datetime import timedelta
from typing import Dict, Any, Optional

from src.custom.connectors.redis import RedisConnector
from src.custom.llm.schemas.redisschema import AskRequest, AskResponse

logger = logging.getLogger(__name__)


class CacheClient:
    """
    Redis-backed cache for exact-match RAG queries.

    Caches responses based on a deterministic hash of request parameters
    to avoid repeated computation for identical requests.
    """

    def __init__(self, connector: RedisConnector, config: Dict[str, Any]):
        """
        Initialize the cache client.

        Args:
            connector: RedisConnector instance used to obtain Redis client.
            config: Cache configuration (supports `ttl_hours`).
        """
        self.redis = connector()
        self.ttl = config["ttl_hours"]

    def _generate_cache_key(self, request: AskRequest) -> str:
        """
        Generate a deterministic Redis cache key for a request.
        """
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "use_hybrid": request.use_hybrid,
            "categories": sorted(request.categories) if request.categories else [],
        }

        key_string = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

        return f"exact_cache:{key_hash}"

    async def find_cached_response(
        self, request: AskRequest
    ) -> Optional[AskResponse]:
        """
        Retrieve a cached response for the given request.

        Returns:
            Cached AskResponse if found, otherwise None.
        """
        try:
            cache_key = self._generate_cache_key(request)
            cached_response = self.redis.get(cache_key)

            if not cached_response:
                logger.info("Redis CACHE MISS")
                return None

            logger.info(f"Redis CACHE HIT → key={cache_key}")
            #logger.info("Cache hit for exact query match")
            return AskResponse(**json.loads(cached_response))

        except Exception:
            logger.exception("Failed to retrieve cached response")
            return None

    async def store_response(
        self, request: AskRequest, response: AskResponse
    ) -> bool:
        """
        Store a response in Redis with configured TTL.

        Returns:
            True if the response was stored successfully.
        """
        try:
            cache_key = self._generate_cache_key(request)

            success = self.redis.set(
                cache_key,
                response.model_dump_json(),
                ex=self.ttl,
            )

            if success:
                logger.info(f"Redis CACHE STORE → key={cache_key}")
            else:
                logger.warning(f"Redis CACHE STORE FAILED → key={cache_key}")

            return bool(success)

        #     return bool(
        #         self.redis.set(
        #             cache_key,
        #             response.model_dump_json(),
        #             ex=self.ttl,
        #         )
        #     )

        except Exception:
            logger.exception("Failed to store response in Redis cache")
            return False
