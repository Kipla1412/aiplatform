from typing import Dict, Any
from .base import CredentialProvider
from src.custom.credentials.localsettings.langfuseconfig import  LangfuseConfig

class LangfuseCredentials(CredentialProvider):
    """
    Provides Langfuse credentials/configuration in a controlled way.
    """

    def __init__(self):
        self.config = LangfuseConfig()

    def get_credentials(self) -> Dict[str, Any]:
        return {
            "enabled": self.config.enabled,
            "public_key": self.config.public_key,
            "secret_key": self.config.secret_key,
            "host": self.config.host,
            "flush_at": self.config.flush_at,
            "flush_interval": self.config.flush_interval,
            "debug": self.config.debug,
        }