import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlencode, quote
from typing import Dict, List, Optional
from ..connectors.arxivconnector import ArxivConnector

logger = logging.getLogger(__name__)

class ArxivMetaExtractor:
    """
    Service Layer - Arxiv Metadata Extractor

    Responsibilities:
    - Uses ArxivConnector HTTP client
    - Applies rate limiting
    - Calls Arxiv API
    - Parses XML into structured JSON metadata

    Does NOT:
    - Store data
    - Handle DB logic
    - Perform PDF parsing
    """

    def __init__(self, connector: ArxivConnector, config: Dict[str, str]):
        """
        Initialize extractor service.

        Args:
            connector (ArxivConnector):
                Established Arxiv HTTP connector

            config (Dict):
                Required keys:
                    base_url (str)
                    rate_limit_delay (int)
                    max_results (int)
                    search_category (str)
                    namespaces (dict)
        """
        self.connector = connector
        self.base_url = config["base_url"]
        self.rate_limit_delay = config["rate_limit_delay"]
        self.max_results = config["max_results"]
        self.category = config["search_category"]
        self.ns = config["namespaces"]
        self._last_request_time: Optional[float] = None

        logger.info(
            "ArxivExtractor initialized | category=%s max_results=%s rate_limit=%ss",
            self.category,
            self.max_results,
            self.rate_limit_delay
        )

    async def _rate_limit(self):
        """
        Ensures Arxiv rate limit policy is respected.
        Arxiv allows:
            - max 1 request every 3 seconds recommended

        Behavior:
        - Checks last request timestamp
        - Sleeps remaining time if needed
        """
        if self._last_request_time is None:
            self._last_request_time = time.time()
            return

        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:

            wait = self.rate_limit_delay - elapsed
            logger.debug("Rate limiting active: sleeping %.2fs", wait)
            await asyncio.sleep(wait)

        self._last_request_time = time.time()

    async def fetch_papers(
        self,
        max_results: Optional[int] = None,
        start: int = 0,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        ) -> List[Dict]:

        """
        Fetch papers from Arxiv and return structured metadata.

        Args:
            max_results: Number of results (defaults to config value)
            start: Pagination start index
            sort_by: submittedDate | lastUpdatedDate | relevance
            sort_order: ascending | descending
            from_date: Filter start date (YYYYMMDD)
            to_date: Filter end date (YYYYMMDD)

        Returns:
            List[Dict]: Parsed paper metadata (id, title, abstract, authors, etc.)

        Raises:
            httpx.HTTPStatusError: API error
            ET.ParseError: If XML parsing fails
        """


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

        logger.info("Requesting Arxiv feed | url=%s", url)

        client = await self.connector()
        response = await client.get(url)
        response.raise_for_status()

        logger.debug("Arxiv response received (%d bytes)", len(response.text))

        return self._parse_xml(response.text)

    def _parse_xml(self, xml_data: str) -> List[Dict]:
        """
        Convert Arxiv XML response into structured JSON.

        Args:
            xml_data (str):
                Raw XML response from Arxiv API

        Returns:
            List[Dict]:
                Parsed metadata list
        """
        logger.info("Parsing Arxiv XML response")

        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", self.ns)

        logger.info("Found %d entries in XML", len(entries))

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

        logger.info("XML parsing completed. Parsed %d papers", len(results))
        return results

    def _get_pdf(self, entry):
        """
        Extracts PDF URL from Arxiv entry node.

        Args:
            entry (xml element):
                Arxiv <entry> element

        Returns:
            str:
                PDF download URL or empty string
        """
        for link in entry.findall("atom:link", self.ns):

            if link.get("type") == "application/pdf":
                pdf_url = link.get("href")
                logger.debug("PDF found: %s", pdf_url)
                return pdf_url
                
        logger.warning("No PDF link found for an entry")       
        return ""
