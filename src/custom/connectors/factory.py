from .rdbms import RDBMSConnector
from .arxivconnector import ArxivConnector
from .apiconnector import JinaConnector
from .opensearch import OpenSearchConnector
from .ollama import OllamaConnector
from .redis import RedisConnector
from .langfuse import LangfuseConnector

class ConnectorFactory:
    """
    The Orchestrator. 
    It just picks the right CLASS based on the category you provide.
    """

    @classmethod
    def get_connector(cls, connector_type: str, config: str):
        """
        Simply returns the connector. 
        The connector will fetch its own credentials.
        """
        if connector_type == "rdbms":
            return RDBMSConnector(config=config)
        elif connector_type == "elasticsearch":
            return ESConnector(config=config)
        elif connector_type == "arxiv":
            return ArxivConnector(config=config)
        elif connector_type == "jina":
            return JinaConnector(config=config)
        elif connector_type == "opensearch":
            return OpenSearchConnector(config=config)
        elif connector_type == "ollama":
            return OllamaConnector(config=config)
        elif connector_type == "redis":
            return RedisConnector(config=config)
        elif connector_type == "langfuse":
            return LangfuseConnector(config=config)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")