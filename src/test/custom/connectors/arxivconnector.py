import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from src.custom.connectors.arxivconnector import ArxivConnector 

pytestmark = pytest.mark.asyncio


SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:atom="http://www.w3.org/2005/Atom">
  <atom:entry>
    <atom:id>http://arxiv.org/abs/1234.5678v1</atom:id>
    <atom:title> Test Paper </atom:title>
    <atom:summary> This is abstract </atom:summary>
    <atom:published>2025-01-01T00:00:00Z</atom:published>

    <atom:author>
      <atom:name>Author One</atom:name>
    </atom:author>
    <atom:author>
      <atom:name>Author Two</atom:name>
    </atom:author>

    <atom:category term="cs.AI"/>
    <atom:category term="cs.CL"/>

    <atom:link type="application/pdf" href="http://arxiv.org/pdf/1234.5678v1.pdf"/>
  </atom:entry>
</feed>
"""


def build_connector():
    config = {
        "base_url": "https://export.arxiv.org/api/query",
        "timeout_seconds": 10,
        "rate_limit_delay": 3,
        "max_results": 25,
        "search_category": "cs.AI",
        "namespaces": {"atom": "http://www.w3.org/2005/Atom"},
    }
    return ArxivConnector(config)


@patch("httpx.AsyncClient")
async def test_fetch_papers_success(mock_client):
    connector = build_connector()

    # mock HTTP client
    instance = mock_client.return_value
    instance.get = AsyncMock()
    instance.get.return_value.status_code = 200
    instance.get.return_value.text = SAMPLE_XML
    instance.get.return_value.raise_for_status = lambda: None

    result = await connector.fetch_papers(max_results=5)

    # ensure client was created
    instance.get.assert_called_once()
    assert len(result) == 1

    paper = result[0]
    assert paper["arxiv_id"] == "1234.5678v1"
    assert paper["title"] == "Test Paper"
    assert paper["abstract"] == "This is abstract"
    assert paper["authors"] == ["Author One", "Author Two"]
    assert paper["categories"] == ["cs.AI", "cs.CL"]
    assert paper["pdf_url"] == "http://arxiv.org/pdf/1234.5678v1.pdf"


@patch("httpx.AsyncClient")
async def test_date_filter_added_to_query(mock_client):
    connector = build_connector()

    instance = mock_client.return_value
    instance.get = AsyncMock()
    instance.get.return_value.status_code = 200
    instance.get.return_value.text = SAMPLE_XML
    instance.get.return_value.raise_for_status = lambda: None

    await connector.fetch_papers(from_date="20250101", to_date="20250131")

    called_url = instance.get.call_args[0][0]
    assert "submittedDate:[202501010000+TO+202501312359]" in called_url


@patch("httpx.AsyncClient")
async def test_rate_limit_waits(mock_client):
    connector = build_connector()
    connector.rate_limit_delay = 1  # speed up test

    instance = mock_client.return_value
    instance.get = AsyncMock()
    instance.get.return_value.status_code = 200
    instance.get.return_value.text = SAMPLE_XML
    instance.get.return_value.raise_for_status = lambda: None

    # first call
    await connector.fetch_papers()

    start = asyncio.get_event_loop().time()

    # second call should sleep
    await connector.fetch_papers()

    elapsed = asyncio.get_event_loop().time() - start
    assert elapsed >= 1

#pytest src/test/custom/connectors/arxivconnector.py -q