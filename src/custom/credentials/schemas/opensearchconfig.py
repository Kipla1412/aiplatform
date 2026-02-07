from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

class OpensearchConfig(BaseSettings):
    """Configuration settings for jina embeddings."""

    model_config = SettingsConfigDict(
        env_prefix="OPENSEARCH_",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )

     # Connection
    host: str
    username: str
    password: str
    port: int = 9200

    # Index settings
    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"
    rrf_pipeline_id: str = "hybrid-rrf-pipeline"


    vector_dimension: int = 1024
    vector_space_type: str = "cosinesimil"

    use_ssl: bool = False
    verify_certs: bool = False
    ssl_show_warn: bool = False

    hybrid_search_size_multiplier: int = 2
