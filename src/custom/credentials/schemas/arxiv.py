from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class ArxivConfig(BaseSettings):
    """
    Simple Arxiv configuration.
    ONLY important values for now.
    No HTTP logic.
    No business logic.
    """

    model_config = SettingsConfigDict(
        env_prefix="ARXIV__",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )

    # Core required values
    base_url: str = "https://export.arxiv.org/api/query"
    timeout_seconds: int = 30
    rate_limit_delay: float = 3.0
    max_results: int = 3
    search_category: str = "cs.AI"

    namespaces: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        "arxiv": "http://arxiv.org/schemas/atom",
    }


