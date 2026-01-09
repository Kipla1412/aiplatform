import logging
from pathlib import Path
from typing import Optional, Dict

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from .exceptions.arxivexceptions import PDFParsingException, PDFValidationError
from .schemas.arxivschema import (
    ParserType,
    PaperFigure,
    PdfContent,
    PaperSection,
    PaperTable,
)

logger = logging.getLogger(__name__)


class ArxivParser:
    """
    Infrastructure Layer — Docling PDF Parser

    This component is responsible ONLY for **low-level PDF processing**.
    Business logic must be handled by the Service Layer.

    Responsibilities:
    - Validate PDF before processing
    - Enforce size and page limits
    - Convert PDF ➝ structured text
    - Return normalized `PdfContent` model
    
    """

    def __init__(self, config: Dict[str, str]):
        """
        Initialize Docling parser engine.

        Args:
            config (Dict[str, str]):
                Expected keys:
                    max_pages (int):
                        Maximum number of pages to process.
                    max_file_size_mb (int):
                        Maximum allowed file size in MB.
                    do_ocr (bool, optional):
                        Enable OCR for scanned PDFs. Disabled by default
                        for performance reasons.
                    do_table_structure (bool, optional):
                        Enable table structure extraction.

        Note:
            OCR is very expensive. Keep it disabled unless required.
        """
        self.max_pages = config["max_pages"]
        self.max_file_size_bytes = config["max_file_size_mb"] * 1024 * 1024

        pipeline_options = PdfPipelineOptions(
            do_table_structure=config.get("do_table_structure", True),
            do_ocr=config.get("do_ocr", False),
        )

        self._converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        self._warmed_up = False

        logger.info(
            "DoclingParser initialized | max_pages=%s max_size=%sMB",
            self.max_pages,
            config["max_file_size_mb"],
        )

    def _warm_up_models(self):
        """
        Warm up Docling model on first use.

        This reduces the first-run latency. Called automatically
        before parsing is executed.
        """
        if not self._warmed_up:
            logger.info("Warming Docling models...")
            self._warmed_up = True

    def _validate_pdf(self, pdf_path: Path):
        """
        Perform strict PDF validation before parsing.

        Validates:
        - File exists
        - File not empty
        - Correct PDF header
        - Size limit
        - Page count limit

        Args:
            pdf_path (Path): Path to the PDF file.

        Raises:
            PDFValidationError:
                If file is missing, corrupted, too large, or exceeds limits.
        """
        if not pdf_path.exists():
            raise PDFValidationError("PDF does not exist")

        size = pdf_path.stat().st_size
        
        if size == 0:
            raise PDFValidationError("Empty PDF file")

        if size > self.max_file_size_bytes:
            raise PDFValidationError("PDF too large")

        with open(pdf_path, "rb") as f:
            if not f.read(8).startswith(b"%PDF-"):
                raise PDFValidationError("Invalid PDF header")

        pdf_doc = pdfium.PdfDocument(str(pdf_path))
        pages = len(pdf_doc)
        pdf_doc.close()

        if pages > self.max_pages:
            raise PDFValidationError(f"Too many pages: {pages}")

    async def parse_pdf(self, pdf_path: Path) -> Optional[PdfContent]:
        """
        Parse and extract structured scientific text using Docling.

        Pipeline:
            1. Validate PDF
            2. Warm model (first time only)
            3. Convert using Docling
            4. Build structured sections
            5. Generate PdfContent response

        Args:
            pdf_path (Path):
                Location of downloaded PDF.

        Returns:
            Optional[PdfContent]:
                Parsed structured document.
                Returns None when validation fails intentionally
                (like too large or too many pages).

        Raises:
            PDFParsingException:
                When an unexpected parsing failure occurs.
        """
        try:
            self._validate_pdf(pdf_path)
            self._warm_up_models()

            result = self._converter.convert(
                str(pdf_path),
                max_num_pages=self.max_pages,
                max_file_size=self.max_file_size_bytes,
            )

            doc = result.document

            sections = []
            current = {"title": "Content", "content": ""}

            for element in doc.texts:
                if hasattr(element, "label") and element.label in (
                    "title",
                    "section_header",
                ):
                    if current["content"].strip():
                        sections.append(
                            PaperSection(
                                title=current["title"],
                                content=current["content"].strip(),
                            )
                        )
                    current = {"title": element.text.strip(), "content": ""}
                else:
                    if hasattr(element, "text") and element.text:
                        current["content"] += element.text + "\n"

            if current["content"].strip():
                sections.append(
                    PaperSection(
                        title=current["title"],
                        content=current["content"].strip(),
                    )
                )

            tables = []
            
            for idx, t in enumerate(getattr(doc, "tables", []), start=1):
                md = ""

                if hasattr(t, "export_to_markdown"):
                        
                    try:
                        md = t.export_to_markdown(doc=doc)  # new API
                    except TypeError:
                        md = t.export_to_markdown()        # fallback

                tables.append(
                    PaperTable(
                        id=str(getattr(t, "uid", None) or f"table_{idx}"),
                        title=getattr(t, "title", None),
                        caption=getattr(t, "caption", None) or "Table extracted from PDF",
                        content=md,
                        metadata={
                            "page": getattr(t, "page_no", None),
                            "bbox": getattr(t, "bbox", None),
                        },
                    )
                )
            


            # ---------- FIGURES ----------
            figures = []
            
            try:
                for idx, f in enumerate(getattr(doc, "figures", []), start=1):
                    figures.append(
                        PaperFigure(
                            id=str(getattr(f, "uid", None) or f"figure_{idx}"),
                            title=getattr(f, "title", None),
                            caption=getattr(f, "caption", None),
                            metadata={
                                "page": getattr(f, "page_no", None),
                                "bbox": getattr(f, "bbox", None),
                            },
                        )
                    )

            except Exception as e:
                logger.error(f"Figure extraction failed: {e}")
                figures = []

            arxiv_id = pdf_path.stem

            return PdfContent(
                sections=sections,
                figures=figures,          
                tables=tables,          
                raw_text=doc.export_to_text(),
                references=[],
                parser_used=ParserType.DOCLING,
                metadata={
                    "source": "docling",
                    "note": "full content extracted including tables & figures",
                    "arxiv_id": arxiv_id,
                },
            )

        except PDFValidationError as e:
            logger.warning(f"Skipping PDF: {e}")
            return None

        except Exception as e:
            logger.error(f"Docling parse failed: {e}")
            raise PDFParsingException(str(e))
