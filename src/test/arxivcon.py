    
import asyncio
from typing import List, Dict
from src.custom.credentials.factory import CredentialFactory
from src.custom.connectors.arxivconnector import ArxivConnector
from src.custom.downloader.arxivpdf import ArxivPDFDownloader

class ArxivService:
    """
    Application Layer
    Hides Infra details
    """

    def __init__(self, mode="arxiv", conn_id=None):
        provider = CredentialFactory.get_provider(mode, conn_id)
        config = provider.get_credentials()

        self.connector = ArxivConnector(config)

    async def get_papers(
        self,
        from_date: str = None,
        to_date: str = None
        ) -> List[Dict]:
        
        print(f"Fetching papers from {from_date} to {to_date}")
        return await self.connector.fetch_papers(
            from_date=from_date,
            to_date=to_date
        )
    
async def run():
    service = ArxivService(mode="arxiv")
    # Use a fixed date range (e.g., last 7 days)
    # This ensures we don't run into future date issues
    from_date = "20241227"  # Example: December 27, 2024
    to_date = "20250102"    # # Example: January 2, 2025
    print(f"Fetching papers from {from_date} to {to_date}")
    papers = await service.get_papers(from_date=from_date, to_date=to_date)
    downloader = ArxivPDFDownloader()

    for p in papers:
        print(f"Paper: {p['title']}")
        print(f"URL: {p['pdf_url']}")

        try:
            pdf_path = await downloader.download(p)
            print(f"Downloaded: {pdf_path}")
        except Exception as e:
            print(f"Failed to download: {e}")

        print("-----")
if __name__ == "__main__":
    asyncio.run(run())
