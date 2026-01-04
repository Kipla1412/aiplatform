from airflow.hooks.base import BaseHook
from .base import CredentialProvider

class AirflowCredentials(CredentialProvider):
    """
    Generic Airflow Credential Provider.
    Unpacks any Airflow Connection into a flat dictionary.
    """
    
    def __init__(self, conn_id: str):
        self.conn_id = conn_id
    
    def get_credentials(self) -> dict:
        # 1. Fetch the connection object from Airflow Metadata
        conn = BaseHook.get_connection(self.conn_id)

        # 2. Build the core dictionary
        # We use dictionary unpacking (**) to merge 'extras' directly
        return {
            "host": conn.host,
            "port": conn.port,
            "user": conn.login,
            "password": conn.password,
            "database": conn.schema, 
            **conn.extra_dejson  # Merges everything else (type, params, etc.)
        }