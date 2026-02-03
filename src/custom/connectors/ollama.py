import httpx
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class OllamaConnector:
    """
    Base asynchronous HTTP connector for Ollama.

    Manages httpx.AsyncClient lifecycle, base URL, and timeout.
    This ONLY handles connection — not LLM logic.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config expected keys:
        - base_url: Ollama host (e.g. http://localhost:11434)
        - timeout_seconds: request timeout
        """
        self.base_url = config["base_url"]
        self.timeout = config.get("timeout_seconds", 30)

        self._client: Optional[httpx.AsyncClient] = None

        logger.info(
            "Ollama HTTP Connector initialized | base_url=%s timeout=%ss",
            self.base_url,
            self.timeout,
        )

    async def __call__(self) -> httpx.AsyncClient:
        """Callable shortcut for connect()."""

        return await self.connect()

    async def _create_client(self) -> httpx.AsyncClient:
        """Create a new async HTTP client for Ollama."""

        logger.info("Creating new Ollama HTTP client session")
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    async def connect(self) -> httpx.AsyncClient:
        """Return active HTTP client, create if not exists."""

        if self._client is None:
            self._client = await self._create_client()
        return self._client

    async def close(self):
        """Close the HTTP client safely."""
        
        if self._client:
            logger.info("Closing Ollama HTTP client session")
            await self._client.aclose()
            self._client = None
