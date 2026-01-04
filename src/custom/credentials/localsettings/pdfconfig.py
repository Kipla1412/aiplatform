from pydantic_settings import BaseSettings, SettingsConfigDict


class ArxivPDFConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARXIV_PDF__",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )

    download_dir: str = "./data/arxiv_pdfs"
    timeout_seconds: int = 30
    rate_limit_delay: float = 3.0
    max_retries: int = 3
    retry_backoff: float = 5.0
