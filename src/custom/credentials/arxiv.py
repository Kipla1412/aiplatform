from typing import Dict
from .base import CredentialProvider
from src.custom.credentials.localsettings.arxivconfig import ArxivConfig


class ArxivCredentials(CredentialProvider):
    """
    Provides Arxiv credentials / important config values
    in a clean & controlled way.
    """

    def __init__(self):
        self.config = ArxivConfig()

    def get_credentials(self) -> Dict[str, str]:
        return {
            "base_url": self.config.base_url,
            "timeout_seconds": self.config.timeout_seconds,
            "rate_limit_delay": self.config.rate_limit_delay,
            "max_results": self.config.max_results,
            "search_category": self.config.search_category,
            "namespaces": self.config.namespaces,
        }
