from typing import Dict, Any
from .base import CredentialProvider
from src.custom.credentials.schemas.ollamaconfig import ollamaconfig

class OllamaCredentials(CredentialProvider):
    """
    Provides Ollama configuration / credentials in a controlled way.
    """

    def __init__(self):
        self.config = ollamaconfig()

    def get_credentials(self) -> Dict[str, Any]:
        return {
            "base_url": self.config.base_url,
            "timeout_seconds": self.config.timeout_seconds,
            "default_model": self.config.default_model,
            "models": self.config.models,
        }
