import asyncio
import logging
import random
from typing import List, Dict, Any

import httpx

from src.custom.embeddings.schema.jinaembedschema import (
    JinaEmbeddingRequest,
    JinaEmbeddingResponse,
)
from src.custom.connectors.apiconnector import JinaConnector

logger = logging.getLogger(__name__)


class JinaEmbeddingsService:
    """
    Jina Embeddings Service.

    Generates embeddings for passages and queries using
    Jina Embeddings v3 via a shared JinaConnector.

    Responsibilities:
    - Build embedding requests
    - Handle retries, backoff, and rate limits
    - Parse and return embedding vectors

    Does NOT:
    - Manage API keys
    - Manage HTTP client lifecycle
    """

    def __init__(self, connector: JinaConnector, config: Dict[str, Any]):
        """
        Initialize embeddings service.

        Parameters
        ----------
        connector : JinaConnector
            Initialized HTTP connector for Jina API.
        config : dict
            Required keys:
            - max_retries (int)
            - base_backoff (float)
        """
        self.connector = connector
        
        self.max_retries: int = config.get("max_retries", 5)
        self.base_backoff: float = config.get("base_backoff", 1.0)
        
        # Embedding config
        self.model = config["model"]
        self.dimensions = config["dimensions"]
        self.tasks = config["tasks"]
        self.batch_size = config.get("batch_size", 100)
   
    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        ) -> dict:
        """
        Execute POST request with retry, exponential backoff,
        jitter, and Retry-After support.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await client.post(url, json=payload)

                # ---- Rate limit handling ----
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after
                        else self.base_backoff * (2 ** (attempt - 1))
                    )

                    logger.warning(
                        "Rate limited (429). Retry %d/%d in %.2fs",
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                response.raise_for_status()
                return response.json()

            except httpx.TimeoutException:
                delay = self._compute_backoff(attempt)
                logger.warning(
                    "Timeout. Retry %d/%d in %.2fs",
                    attempt,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

            except httpx.HTTPStatusError as e:
                status = e.response.status_code

                # Retry only on 5xx
                if 500 <= status < 600:
                    delay = self._compute_backoff(attempt)
                    logger.warning(
                        "Server error %s. Retry %d/%d in %.2fs",
                        status,
                        attempt,
                        self.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Non-retriable HTTP error: %s", status)
                    raise

            except httpx.HTTPError as e:
                delay = self._compute_backoff(attempt)
                logger.warning(
                    "Network error. Retry %d/%d in %.2fs | %s",
                    attempt,
                    self.max_retries,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

        raise RuntimeError("Max retries exceeded for embeddings request")

    def _compute_backoff(self, attempt: int) -> float:
        """
        Compute exponential backoff with jitter.
        """
        base = self.base_backoff * (2 ** (attempt - 1))
        jitter = random.uniform(0, base * 0.3)
        return base + jitter

    
    async def embed_passages(
        self,
        texts: List[str],
        batch_size: int = 100,
        ) -> List[List[float]]:
        """
        Generate embeddings for text passages.

        Parameters
        ----------
        texts : list[str]
            Text passages to embed.
        batch_size : int
            Number of passages per API request.

        Returns
        -------
        list[list[float]]
            Embedding vectors.
        """
        client = await self.connector.connect()
        embeddings: List[List[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            payload = JinaEmbeddingRequest(
                model=self.model,
                task=self.tasks["passage"],
                dimensions=self.dimensions,
                input=batch,
                )

            response_json = await self._post(
                client,
                "/embeddings",
                payload.model_dump(),
            )

            result = JinaEmbeddingResponse(**response_json)
            embeddings.extend([item["embedding"] for item in result.data])

            logger.debug("Embedded %d passages", len(batch))

        logger.info("Generated %d passage embeddings", len(embeddings))
        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a search query.

        Parameters
        ----------
        query : str
            Query text.

        Returns
        -------
        list[float]
            Query embedding vector.
        """
        client = await self.connector.connect()

        payload = JinaEmbeddingRequest(
            model=self.model,
            task=self.tasks["query"],
            dimensions=self.dimensions,
            input=[query],
        )

        response_json = await self._post(
            client,
            "/embeddings",
            payload.model_dump(),
        )

        result = JinaEmbeddingResponse(**response_json)
        return result.data[0]["embedding"]
