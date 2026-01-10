import httpx
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class ArxivConnector:
    """
    Infrastructure Layer - Arxiv HTTP Connector

    Responsible only for:
    - Creating reusable async HTTP connection
    - Managing client lifecycle
    - Returning connection object to services
    """

    def __init__(self, config: Dict[str, str]):
        """
        Initialize ArxivConnector.

        Args:
            config (Dict):
                Required keys:
                    base_url (str): Arxiv API endpoint
                    timeout_seconds (int): HTTP timeout in seconds
        """
        self.base_url = config["base_url"]
        self.timeout = config["timeout_seconds"]
        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "ArxivConnector initialized | base_url=%s timeout=%ss",
            self.base_url,
            self.timeout
        )

    
    async def __call__(self) -> httpx.AsyncClient:
        """
        Callable version of connect().

        Allows writing:
            client = await connector()

        Returns:
            httpx.AsyncClient
        """
        return await self.connect()

    async def connect(self) -> httpx.AsyncClient:
        """
        Creates (if needed) and returns async HTTP client.

        This method ensures:
        - Reusable persistent connection
        - Lazy initialization
        - Centralized client management

        Returns:
            httpx.AsyncClient: Active HTTP client instance
        """
        if self._client is None:
            logger.info("Creating new HTTP client session for Arxiv")
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout
            )

        else:
            logger.debug("Reusing existing Arxiv HTTP client session")
        
        return self._client


    async def close(self):
        """
        Gracefully closes HTTP connection.

        When to use:
        - Application shutdown
        - Airflow teardown
        - Service cleanup
        """
        if self._client:

            logger.info("Closing Arxiv HTTP client session")
            await self._client.aclose()
            self._client = None
        
        else:
            logger.debug("Close called but no HTTP client session exists")
