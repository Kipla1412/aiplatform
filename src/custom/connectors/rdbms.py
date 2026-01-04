from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from ..credentials.factory import CredentialFactory

class RDBMSConnector:
    def __init__(self, config):
        """
        Initializes the connector by fetching credentials internally.
        """
        self.config = config
        self._engine = None
        self._connection = None

    def __call__(self):
        self.connect()
        return self._connection

    def connect(self) -> None:
        """ Create SQLAlchemy engine and test connection. """
        
        # Extract basic URL parts from the dictionary
        # Note: mapping 'user' from dict to 'username' for SQLAlchemy URL
        url_params = {
            "drivername": self.config.get("db_type") or self.config.get("type"),
            "username": self.config.get("user"),
            "password": self.config.get("password"),
            "host": self.config.get("host"),
            "port": self.config.get("port"),
            "database": self.config.get("database"),
        }

        # Create the connection URL (filtering out None values)
        connection_url = URL.create(**{k: v for k, v in url_params.items() if v is not None})

        # 4. Create Engine
        self._engine = create_engine(connection_url)
        
        # 5. Test Connection
        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        self._connection = self._engine
