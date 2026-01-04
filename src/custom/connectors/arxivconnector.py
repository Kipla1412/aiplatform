import httpx
from typing import Optional, Dict


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
            self._client = httpx.AsyncClient(timeout=self.timeout)

        return self._client

    async def __call__(self) -> httpx.AsyncClient:
        """
        Callable version of connect().

        Allows writing:
            client = await connector()

        Returns:
            httpx.AsyncClient
        """
        return await self.connect()

    async def close(self):
        """
        Gracefully closes HTTP connection.

        When to use:
        - Application shutdown
        - Airflow teardown
        - Service cleanup
        """
        if self._client:
            await self._client.aclose()
            self._client = None
