from .airflow import AirflowCredentials
from .arxiv import ArxivCredentials

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
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'airflow' or 'local'.")