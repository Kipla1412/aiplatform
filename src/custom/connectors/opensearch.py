import logging
from typing import Optional, Dict, Any
from opensearchpy import OpenSearch

logger = logging.getLogger(__name__)


class OpenSearchConnector:
    """
    Infrastructure Layer — OpenSearch Connector

    Responsibilities:
    - Create and reuse OpenSearch client
    - Manage connection lifecycle
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenSearch connector.

        Expected config keys:
        ---------------------
        host : str
        port : int
        use_ssl : bool
        """
        self.host = config["host"]
        self.port = config["port"]
        self.use_ssl = config.get("use_ssl", False)

        self._client: Optional[OpenSearch] = None

        logger.info(
            "OpenSearchConnector initialized | host=%s port=%s ssl=%s",
            self.host,
            self.port,
            self.use_ssl,
        )

    def connect(self) -> OpenSearch:
        """
        Return a reusable OpenSearch client.

        Creates the client lazily on first call.
        """
        if self._client is None:
            logger.info("Creating OpenSearch client")
            self._client = OpenSearch(
                hosts=[{"host": self.host, "port": self.port}],
                use_ssl=self.use_ssl,
                verify_certs=False,
                ssl_show_warn=False,
            )
        return self._client

    def close(self):
        """
        Reset OpenSearch connection reference.

        (OpenSearch client does not require explicit close.)
        """
        logger.info("OpenSearch connector closed")
        self._client = None
