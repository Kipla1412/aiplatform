from typing import Dict, Any
from .base import CredentialProvider
from src.custom.credentials.schemas.jinaconfig import JinaConfig

class JinaCredentials(CredentialProvider):
    """
    Provides Jina Embeddings configuration
    in a clean, structured, and controlled way.
    """

    def __init__(self):
        self.config = JinaConfig()

    def get_credentials(self) -> Dict[str, Any]:
        
        return {
            "base_url": self.config.base_url,
            "api_key": self.config.api_key,
            "timeout_seconds": self.config.timeout_seconds,
            "max_retries": self.config.max_retries,
            "base_backoff": self.config.base_backoff,
            "model": self.config.model,
            "dimensions": self.config.dimensions,
            "batch_size": self.config.batch_size,
            "tasks": {
                "passage": self.config.tasks.passage,
                "query": self.config.tasks.query,
            },
        }