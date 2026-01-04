import pytest
from src.custom.connectors.arxivconnector import ArxivConnector


config = {
    "base_url": "https://export.arxiv.org/api/query",
    "timeout_seconds": 5,
}


@pytest.mark.asyncio
async def test_connector_creates_client():
    connector = ArxivConnector(config)

    client = await connector.connect()

    assert client is not None
    assert client.timeout is not None


@pytest.mark.asyncio
async def test_connector_reuses_same_client():
    connector = ArxivConnector(config)

    c1 = await connector.connect()
    c2 = await connector()

    assert c1 is c2   # Same object → singleton behavior


@pytest.mark.asyncio
async def test_connector_close_creates_new_client_after():
    connector = ArxivConnector(config)

    c1 = await connector.connect()
    await connector.close()

    c2 = await connector.connect()

    assert c2 is not None
    assert c1 is not c2   # New object created
