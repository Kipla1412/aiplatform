from pydantic import BaseModel
from typing import Optional


class RedisConfig(BaseModel):
    """
    Purpose:
        Holds Redis connection details (not business logic).
    """

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    ttl: int = 2
