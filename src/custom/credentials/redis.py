from typing import Dict, Any
from .base import CredentialProvider
from src.custom.credentials.schemas.redisconfig import RedisConfig


class RedisCredentials(CredentialProvider):
    """
    Provides Redis connection credentials in a controlled way.
    """

    def __init__(self):
        self.config = RedisConfig()

    def get_credentials(self) -> Dict[str, Any]:
        return {
            "host": self.config.host,
            "port": self.config.port,
            "db": self.config.db,
            "password": self.config.password,
            "ssl": self.config.ssl,
            "ttl_hours": self.config.ttl
        }
