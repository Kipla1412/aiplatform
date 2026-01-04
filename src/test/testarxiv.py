import asyncio
from src.custom.connectors.arxivconnector import ArxivConnector
from src.custom.downloader.arxivdownloader import ArxivExtractor

# config = {
#     "base_url": "https://export.arxiv.org/api/query",
#     "timeout_seconds": 10,
# }

# async def main():
#     connector = ArxivConnector(config)

#     # get client
#     client1 = await connector.connect()
#     print("Client 1:", client1)

#     # call again to ensure same instance (reuse check)
#     client2 = await connector()
#     print("Client 2:", client2)

#     print("Same object?", client1 is client2)

#     # close connection
#     await connector.close()
#     print("Closed successfully")

# asyncio.run(main())

from src.custom.credentials.factory import CredentialFactory
from src.custom.connectors.arxivconnector import ArxivConnector
from src.custom.downloader.arxivmetaextractor import ArxivMetaExtractor
from src.custom.downloader.arxivdownloader import ArxivPDFDownloader

class ArxivService:
    """
    Application Layer
    """

    def __init__(self, mode="arxivlocal", conn_id=None):
        provider = CredentialFactory.get_provider(mode, conn_id)
        self.config = provider.get_credentials()

        self.connector = ArxivConnector(self.config)
        self.extractor = ArxivMetaExtractor(self.connector, self.config)
        self.downloader = ArxivPDFDownloader()
        
    async def get_latest_papers(self, limit=5):
        return await self.extractor.fetch_papers(max_results=limit)

    async def download_pdf(self, paper):
        return await self.downloader.download(paper)

    async def close(self):
        await self.connector.close()
        await self.downloader.close()

async def main():
    service = ArxivService(mode="arxivlocal")

    papers = await service.get_latest_papers(limit=3)

    for p in papers:
        print("\n--------------------------------")
        print("ID:", p["arxiv_id"])
        print("Title:", p["title"])
        print("Authors:", ", ".join(p["authors"]))
        print("PDF:", p["pdf_url"])

        path = await service.download_pdf(p)
        print("Downloaded:", path)

    await service.close()
asyncio.run(main())