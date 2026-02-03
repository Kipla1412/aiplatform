from pydantic import BaseModel
from typing import Optional


class RedisConfig(BaseModel):
    """
    Purpose:
        Configuration schema for redis connections.
    """
    host: str
    port: int
    db: int 
    password: Optional[str]
    ssl: bool 
