import logging
from typing import Optional, Dict, Any

from langfuse import Langfuse

logger = logging.getLogger(__name__)


class LangfuseConnector:
    """
    Connector responsible for initializing and managing the Langfuse client.

    This class ensures:
    - Safe initialization
    - App does not crash if Langfuse fails
    - Centralized client management
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._client: Optional[Langfuse] = None
        self._initialize()

    def __call__(self) -> Optional[Langfuse]:
        """
        Allows the connector instance to be called like a function
        to retrieve the Langfuse client.
        """
        return self._client

    def _initialize(self) -> None:
        """Initialize Langfuse client if enabled."""
        if not self.config.get("enabled"):
            logger.info("Langfuse is disabled via configuration")
            return

        if not self.config.get("public_key") or not self.config.get("secret_key"):
            logger.warning("Langfuse keys missing — tracing disabled")
            return

        try:
            self._client = Langfuse(
                # public_key=self.config.public_key,
                # secret_key=self.config.secret_key,
                # host=self.config.host,
                # flush_at=self.config.flush_at,
                # flush_interval=self.config.flush_interval,
                # debug=self.config.debug,
                # )
               
                # public_key=self.config.get("public_key"),
                # secret_key=self.config.get("secret_key"),
                # host=self.config.get("host"),
                # flush_at=self.config.get("flush_at", 1),
                # flush_interval=self.config.get("flush_interval", 1),
                # debug=self.config.get("debug", False),
                public_key=self.config["public_key"],
                secret_key=self.config["secret_key"],
                host=self.config["host"],
                flush_at=self.config["flush_at"],
                flush_interval=self.config["flush_interval"],
                debug=self.config["debug"],
            )
            logger.info(f"Langfuse initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Langfuse: {e}")
            self._client = None

    def get_client(self) -> Optional[Langfuse]:
        """Return the Langfuse client instance if available."""
        return self._client

    def flush(self) -> None:
        """Flush any pending Langfuse events."""
        if self._client:
            try:
                self._client.flush()
            except Exception as e:
                logger.error(f"Error flushing Langfuse events: {e}")

    def shutdown(self) -> None:
        """Shutdown Langfuse client safely."""
        if self._client:
            try:
                self._client.flush()
                self._client.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down Langfuse: {e}")