import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, List

from .exceptions.arxivexceptions import PDFParsingException, PDFValidationError
from .schemas.arxivschema import (
    ParserType,
    PaperFigure,
    PdfContent,
    PaperSection,
    PaperTable,
)
from .arxivparser import ArxivParser

logger = logging.getLogger(__name__)


class PDFParserService:
    """
    Service Layer wrapper around DoclingParser.

    Purpose:
    - Validate input PDF
    - Delegate parsing to Docling
    - Provide consistent logging & error handling

    Notes:
    - Does NOT download PDFs
    - Does NOT store data
    - Only controls parsing workflow
    """

    def __init__(self, config: Dict):
        """
        Initialize parser service.

        Parameters
        ----------
        config : Dict
            Expected structure:
            { "docling": {...parser settings...} }
        """
        self.config = config["docling"]
        self.input_dir = Path(self.config["pdf_dir"])

        if not self.input_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {self.input_dir}")

        self.parser = ArxivParser(self.config) 
        self.max_concurrency = self.config["max_concurrency"]
        self.semaphore = asyncio.Semaphore(self.max_concurrency)


    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse a PDF and return structured content.

        Parameters
        ----------
        pdf_path : Path
            Local PDF file path.

        Returns
        -------
        Optional[PdfContent]
            Parsed PDF content or None (when intentionally skipped)

        Raises
        ------
        PDFValidationError : Invalid or missing file
        PDFParsingException : Parsing failure
        """
        if not pdf_path.exists():
            raise PDFValidationError(f"PDF not found: {pdf_path}")
        
        async with self.semaphore:

            try:
                result = await self.parser.parse_pdf(pdf_path)

                if result:
                    logger.info("Parsed PDF: %s", pdf_path.name)
                    return result

                raise PDFParsingException("Docling returned no result")

            except (PDFValidationError, PDFParsingException):
                raise
            except Exception as e:
                logger.error("Unexpected parse failure: %s", e)
                raise PDFParsingException(str(e))

    async def parse_all_pdfs(self) -> List[PdfContent]:
        """
        Parse all PDFs in configured directory in parallel.

        Returns
        -------
        List[PdfContent]
            Successfully parsed documents only
        """
        if not self.input_dir.exists():
            logger.error("PDF directory missing: %s", self.input_dir) 
            return []

        pdf_files = list(self.input_dir.glob("*.pdf")) + list(self.input_dir.glob("*.PDF"))

        if not pdf_files:
            logger.warning("No PDFs found in %s", self.input_dir)
            return []

        logger.info("Starting batch parse: %d files (Concurrency: %d)", 
                    len(pdf_files), self.max_concurrency)

        tasks = [self.parse_pdf(pdf) for pdf in pdf_files]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        parsed: List[PdfContent] = []

        for file, result in zip(pdf_files, results):
            if isinstance(result, Exception):
                logger.error("Failed parsing %s -> %s", file.name, result)
            elif result:
                parsed.append(result)

        logger.info("Parsed %d/%d PDFs successfully", len(parsed), len(pdf_files))
        return parsed
