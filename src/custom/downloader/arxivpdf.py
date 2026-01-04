from pathlib import Path
from typing import Optional
import asyncio
import httpx
import time
import logging

from ..credentials.localsettings.pdfconfig import ArxivPDFConfig


logger = logging.getLogger(__name__)


class ArxivPDFDownloader:
    """
    Infra Layer
    Downloads arXiv PDFs with:
    - Config via ArxivPDFConfig
    - Logging
    - Rate limiting
    - Retry with backoff
    - Cached downloads
    """

    def __init__(self, config: Optional[ArxivPDFConfig] = None):
        self.config = config or ArxivPDFConfig()

        self.download_dir = Path(self.config.download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        self.timeout = self.config.timeout_seconds
        self.rate_limit_delay = self.config.rate_limit_delay
        self.max_retries = self.config.max_retries
        self.retry_backoff = self.config.retry_backoff

        self._last_request_time: Optional[float] = None

        logger.info(
            f"ArxivPDFDownloader initialized | dir={self.download_dir} "
            f"timeout={self.timeout}s rate_limit={self.rate_limit_delay}s"
        )

    async def _rate_limit(self):
        """Respect arXiv polite usage rule."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self.rate_limit_delay:
                wait = self.rate_limit_delay - elapsed
                logger.debug(f"Rate limiting active, sleeping {wait:.2f}s")
                await asyncio.sleep(wait)

        self._last_request_time = time.time()

    async def download(self, paper: dict, force: bool = False) -> Optional[Path]:
        """
        Download PDF for the given arXiv paper dict.

        Expected dict keys:
            - arxiv_id
            - pdf_url

        Returns:
            Path or None
        """

        pdf_url = paper.get("pdf_url")
        arxiv_id = paper.get("arxiv_id")

        if not pdf_url or not arxiv_id:
            logger.error("Missing pdf_url or arxiv_id in paper dict")
            return None

        pdf_path = self.download_dir / f"{arxiv_id}.pdf"

        # Cached
        if pdf_path.exists() and not force:
            logger.info(f"PDF already exists, using cache: {pdf_path.name}")
            return pdf_path

        await self._rate_limit()

        logger.info(f"Downloading PDF for {arxiv_id}")

        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream("GET", pdf_url) as response:
                        response.raise_for_status()

                        with open(pdf_path, "wb") as file:
                            async for chunk in response.aiter_bytes():
                                file.write(chunk)

                logger.info(f"Successfully downloaded: {pdf_path.name}")
                return pdf_path

            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(
                        f"Download failed for {arxiv_id} after "
                        f"{attempt} attempts | error={e}"
                    )
                    return None

                wait = self.retry_backoff * attempt
                logger.warning(
                    f"Download attempt {attempt}/{self.max_retries} failed "
                    f"for {arxiv_id} | retrying in {wait}s | error={e}"
                )
                await asyncio.sleep(wait)

        return None
