import pytest
import httpx
from unittest.mock import AsyncMock, patch

from src.custom.connectors.apiconnector import JinaConnector


def test_jina_connector_init_success():
    config = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "test_api_key",
        "timeout_seconds": 30,
    }

    connector = JinaConnector(config)

    assert connector.base_url == "https://api.jina.ai/v1"
    assert connector.api_key == "test_api_key"
    assert connector.timeout == 30
    assert connector.headers["Authorization"] == "Bearer test_api_key"

# Missing API key

def test_jina_connector_missing_api_key():
    config = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "",
        "timeout_seconds": 30,
    }

    with pytest.raises(ValueError):
        JinaConnector(config)

# connect() creates client

@pytest.mark.asyncio
async def test_connect_creates_client():
    config = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "test_api_key",
        "timeout_seconds": 30,
    }

    connector = JinaConnector(config)

    mock_client = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client) as mock_cls:
        client = await connector.connect()

        assert client is mock_client
        mock_cls.assert_called_once()

# connect() reuses same client

@pytest.mark.asyncio
async def test_connect_reuses_existing_client():
    config = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "test_api_key",
        "timeout_seconds": 30,
    }

    connector = JinaConnector(config)
    mock_client = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client):
        client1 = await connector.connect()
        client2 = await connector.connect()

        assert client1 is client2

# __call__ delegates to connect

@pytest.mark.asyncio
async def test_callable_returns_client():
    config = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "test_api_key",
        "timeout_seconds": 30,
    }

    connector = JinaConnector(config)
    mock_client = AsyncMock()

    with patch("httpx.AsyncClient", return_value=mock_client):
        client = await connector()
        assert client is mock_client


# close() closes client

@pytest.mark.asyncio
async def test_close_client():
    config = {
        "base_url": "https://api.jina.ai/v1",
        "api_key": "test_api_key",
        "timeout_seconds": 30,
    }

    connector = JinaConnector(config)
    mock_client = AsyncMock()
    connector._client = mock_client

    await connector.close()

    mock_client.aclose.assert_awaited_once()
    assert connector._client is None
