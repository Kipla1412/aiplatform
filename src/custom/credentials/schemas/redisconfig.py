from pydantic import BaseModel
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class RedisConfig(BaseSettings):
    """
    Purpose:
        Holds Redis connection details (not business logic).
    """
    model_config = SettingsConfigDict(
        env_prefix="REDIS__",
        env_file=[".env"],
        extra="ignore",
        case_sensitive=False,
    )

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ttl: int = 2
