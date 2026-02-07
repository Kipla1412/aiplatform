from typing import Dict, Any
from .base import CredentialProvider
from src.custom.credentials.schemas.opensearchconfig import OpensearchConfig

class OpenSearchCredentials(CredentialProvider):
    """
    Provides Jina Embeddings configuration
    in a clean, structured, and controlled way.
    """

    def __init__(self):
        self.config = OpensearchConfig()

    def get_credentials(self) -> Dict[str, Any]:
        return {
            "host": self.config.host,
            "username": self.config.username,
            "password": self.config.password,
            "port": self.config.port,
            "index_name": self.config.index_name,
            "chunk_index_suffix": self.config.chunk_index_suffix,
            "rrf_pipeline_id": self.config.rrf_pipeline_id,
            "vector_dimension": self.config.vector_dimension,
            "vector_space_type": self.config.vector_space_type,
            "use_ssl": self.config.use_ssl,
            "verify_certs": self.config.verify_certs,
            "ssl_show_warn": self.config.ssl_show_warn,
            "hybrid_search_size_multiplier": self.config.hybrid_search_size_multiplier,
        }