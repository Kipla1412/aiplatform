from .airflow import AirflowCredentials
from .arxiv import ArxivCredentials
from .ollama import OllamaCredentials
from .redis import RedisCredentials
from .langfuse import LangfuseCredentials
from .jina import JinaCredentials
from .opensearch import OpenSearchCredentials

class CredentialFactory:
    """
    Decides which 'Source' to use for credential management.
    Uses 'conn_id' as the master key for both Airflow and Local YAML.
    """
    @staticmethod
    def get_provider(mode: str, conn_id: str = None):
        if mode == "airflow":
            return AirflowCredentials(conn_id)
        
        # elif mode == "local":
        #     return LocalCredentials(conn_id)

        elif mode == "arxivlocal":
            return ArxivCredentials()

        elif mode == "ollama":
            return OllamaCredentials()

        elif mode == "redis":
            return RedisCredentials()

        elif mode == "langfuse":
            return LangfuseCredentials() 

        elif mode == "jina":
            return JinaCredentials() 

        elif mode == "opensearch":
            return OpenSearchCredentials()      
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'airflow' or 'local'.")