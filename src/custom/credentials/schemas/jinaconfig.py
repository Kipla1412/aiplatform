from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from pydantic import Field
from dotenv import load_dotenv

class JinaTaskConfig(BaseSettings):
    """Nested task configuration for Jina embedding API."""

    passage: str = Field(default="retrieval.passage")
    query: str = Field(default="retrieval.query")



class JinaConfig(BaseSettings):
    """Configuration settings for jina embeddings."""

    model_config = SettingsConfigDict(
        env_prefix="JINA_",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )

    # Core API connection
    base_url: str
    api_key: str

    # Networking / reliability
    timeout_seconds: int = 30
    max_retries: int = 5
    base_backoff: float = 1.0

    # Model settings
    model: str 
    dimensions: int = 1024
    batch_size: int = 50

     # Task configuration (nested)
    tasks: JinaTaskConfig = JinaTaskConfig()

