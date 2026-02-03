import logging
import redis
from src.custom.connectors.schemas.redisconfig import RedisConfig

logger = logging.getLogger(__name__)


class RedisConnector:

    """
    RedisConnector
    =================
    Purpose:
        Provides a reusable connector for Redis that:
        - Creates a Redis client using configuration values
        - Verifies connectivity using a health check (PING)
        - Returns a ready-to-use Redis client instance

    Usage:
        connector = RedisConnector(config_dict)
        redis_client = connector()   # returns redis.Redis instance
    """

    def __init__(self, config: dict):
        
        """
        Initialize the RedisConnector with configuration.

        Args:
            config (dict):
                Dictionary containing Redis connection details.
                Expected keys:
                    - host (str): Redis host
                    - port (int): Redis port
                    - db (int): Redis database index
                    - password (str | None): Password for authentication
                    - ssl (bool): Enable SSL/TLS connection
        """
        self.config = RedisConfig(**config)
        self._client = None

    def __call__(self) -> redis.Redis:
        """
        Make the connector callable.

        Purpose:
            Allows the instance to be called like a function to
            return a connected Redis client.

        Returns:
            redis.Redis: Active Redis client instance
        """
        self.connect()
        return self._client

    def connect(self) -> None:

        """
        Establish a connection to Redis and perform a health check.

        Process:
            1. Create a Redis client using provided configuration
            2. Ping the Redis server to verify connectivity
            3. Raise an error if the connection fails

        Raises:
            ConnectionError:
                If Redis is unreachable or authentication fails
        """
        logger.info(
            f"Attempting to connect to Redis at {self.config.host}:{self.config.port}"
        )

        # Create Redis client
        self._client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            ssl=self.config.ssl,
            decode_responses=True,  # Return strings instead of bytes
        )

        try:
            # Health check to verify Redis availability
            self._client.ping()
        except redis.ConnectionError as e:
            logger.error("Could not connect to Redis", exc_info=e)
            raise ConnectionError("Redis connection failed") from e

        logger.info("Redis connection verified successfully.")
