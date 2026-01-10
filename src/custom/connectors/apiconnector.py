import httpx
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class JinaConnector:
    """
    Base asynchronous HTTP connector.

    Provides a reusable wrapper around `httpx.AsyncClient`
    for managing connection lifecycle, headers, base URL,
    and timeout configuration.

    Intended to be extended by concrete API connectors.
    """

    def __init__(self, config: Dict[str, str]):
        """
        Initialize the HTTP connector.

        Parameters
        
        base_url : str
            Base URL for all HTTP requests.
        timeout : int, optional
            Request timeout in seconds (default: 30).
        headers : dict[str, str], optional
            Default headers applied to all requests.
        """
        self.base_url= config["base_url"]
        self.api_key = config["api_key"]

        if not self.api_key:
            raise ValueError("Missing api_key for Jina API")

        self.timeout = config["timeout_seconds"]
        
        self.headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "HTTP Connector initialized | base_url=%s timeout=%ss",
            self.base_url,
            self.timeout,
        )

    async def __call__(self) -> httpx.AsyncClient:
        """
        Callable shortcut for `connect()`.

        Returns

        httpx.AsyncClient
            Active HTTP client.
        """
        return await self.connect()


    async def _create_client(self) -> httpx.AsyncClient:
        """
        Create a new asynchronous HTTP client.

        Returns
        
        httpx.AsyncClient
            Configured async HTTP client instance.
        """
        logger.info("Creating new HTTP client session")
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=self.timeout,
        )

    async def connect(self) -> httpx.AsyncClient:
        """
        Get an active HTTP client.

        Creates a new client if one does not already exist,
        otherwise returns the existing instance.

        Returns
        
        httpx.AsyncClient
            Active HTTP client.
        """
        if self._client is None:
            self._client = await self._create_client()
        return self._client

    async def close(self):
        """
        Close the HTTP client and release resources.

        Safe to call multiple times.
        """
        if self._client:
            logger.info("Closing HTTP client session")
            await self._client.aclose()
            self._client = None
