import asyncio
import time
import httpx
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime
from urllib.parse import urlencode, quote
from ..credentials.factory import CredentialFactory

class ArxivConnector:
    """
    Arxiv Connector (Infra Layer)
    
    - Accepts config dict
    - Lazily creates reusable async HTTP client
    - Applies rate limiting
    - Returns JSON-like metadata (not XML)
    """

    def __init__(self, config: Dict[str, str]):
        
        self.config = config
        self.base_url = config["base_url"]
        self.timeout = config["timeout_seconds"]
        self.rate_limit_delay = config["rate_limit_delay"]
        self.max_results = config["max_results"]
        self.category = config["search_category"]
        self.ns = config["namespaces"]
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_time: Optional[float] = None

    def __call__(self):
        
        return self

    async def connect(self):
        """Create reusable HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)

    async def _rate_limit(self):
        """Respect arXiv 3s rule"""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                await asyncio.sleep(self.rate_limit_delay - elapsed)

        self._last_request_time = time.time()


    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,   # YYYYMMDD
        to_date: Optional[str] = None,     # YYYYMMDD
        ) -> List[Dict]:


        """
        Fetch papers with advanced filtering options.
        
        Args:
            max_results: Maximum number of results to return
            start: Starting index for pagination
            sort_by: Sort field (submittedDate, lastUpdatedDate, relevance)
            sort_order: Sort direction (ascending, descending)
            from_date: Start date in YYYYMMDD format
            to_date: End date in YYYYMMDD format
            
        Returns:
            List of paper metadata dictionaries
        """

        await self.connect()
        await self._rate_limit()

        if max_results is None:
            max_results = self.max_results

        search_query = f"cat:{self.category}"

        if from_date or to_date:
            date_from = f"{from_date}0000" if from_date else "*"
            date_to = f"{to_date}2359" if to_date else "*"
            search_query += f"+AND+submittedDate:[{date_from}+TO+{date_to}]"

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        safe = ":+[]*"      
        url = f"{self.base_url}?{urlencode(params, quote_via=quote, safe=safe)}"

        response = await self._client.get(url)
        response.raise_for_status()

        return self._parse_xml(response.text)


    def _parse_xml(self, xml_data: str) -> List[Dict]:
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", self.ns)

        results = []

        for entry in entries:
            paper = {
                "arxiv_id": entry.find("atom:id", self.ns).text.split("/")[-1],
                "title": entry.find("atom:title", self.ns).text.strip(),
                "abstract": entry.find("atom:summary", self.ns).text.strip(),
                "published_date": entry.find("atom:published", self.ns).text,
                "authors": [
                    a.find("atom:name", self.ns).text
                    for a in entry.findall("atom:author", self.ns)
                ],
                "categories": [
                    c.get("term")
                    for c in entry.findall("atom:category", self.ns)
                ],
                "pdf_url": self._get_pdf(entry),
            }

            results.append(paper)

        return results

    def _get_pdf(self, entry):
        for link in entry.findall("atom:link", self.ns):
            if link.get("type") == "application/pdf":
                return link.get("href")
        return ""
