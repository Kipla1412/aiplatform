import pytest

from src.custom.downloader.arxivmetaextractor import ArxivMetaExtractor
from src.custom.connectors.arxivconnector import ArxivConnector


FAKE_XML = """
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title> Test Paper Title </title>
    <summary> This is a test abstract. </summary>
    <published>2024-01-01T00:00:00Z</published>

    <author><name>Alice</name></author>
    <author><name>Bob</name></author>

    <category term="cs.AI"/>

    <link href="http://arxiv.org/pdf/1234.5678v1" type="application/pdf"/>
  </entry>
</feed>
"""


config = {
    "base_url": "https://export.arxiv.org/api/query",
    "timeout_seconds": 10,
    "rate_limit_delay": 0,
    "max_results": 5,
    "search_category": "cs.AI",
    "namespaces": {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    },
}


@pytest.mark.asyncio
async def test_fetch_papers_parses_xml_correctly(monkeypatch):
    """
    Unit test for ArxivExtractor
    Mocks HTTP call and ensures XML parsing works properly.
    """

    connector = ArxivConnector(config)
    extractor = ArxivExtractor(connector, config)

    # ---------- Fake Response ----------
    class FakeResponse:
        status_code = 200
        text = FAKE_XML

        def raise_for_status(self):
            pass

    async def fake_get(url):
        return FakeResponse()

    # ---------- Patch Client ----------
    client = await connector.connect()
    monkeypatch.setattr(client, "get", fake_get)

    # ---------- Call ----------
    papers = await extractor.fetch_papers(max_results=1)

    # ---------- Assertions ----------
    assert len(papers) == 1

    paper = papers[0]

    assert paper["arxiv_id"] == "1234.5678v1"
    assert paper["title"] == "Test Paper Title"
    assert paper["abstract"] == "This is a test abstract."
    assert paper["published_date"] == "2024-01-01T00:00:00Z"
    assert paper["authors"] == ["Alice", "Bob"]
    assert paper["categories"] == ["cs.AI"]
    assert paper["pdf_url"] == "http://arxiv.org/pdf/1234.5678v1"
