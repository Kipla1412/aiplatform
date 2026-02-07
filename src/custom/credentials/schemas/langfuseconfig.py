from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv

load_dotenv()

class LangfuseConfig(BaseSettings):
    """Configuration settings for Langfuse observability."""

    model_config = SettingsConfigDict(
        env_prefix="LANGFUSE_",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )

    enabled: bool = True
    public_key: str | None = None
    secret_key: str | None = None
    host: str = "https://cloud.langfuse.com"
    flush_at: int = 10
    flush_interval: float = 2.0
    debug: bool = False
