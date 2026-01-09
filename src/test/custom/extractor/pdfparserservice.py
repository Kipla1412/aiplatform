import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.custom.extractors.pdfparserservice import PDFParserService
from src.custom.extractors.arxivparser import ArxivParser
from src.custom.extractors.exceptions.arxivexceptions import PDFValidationError, PDFParsingException
from src.custom.extractors.schemas.arxivschema import PdfContent, ParserType


@pytest.fixture
def config(tmp_path):
    return {
        "docling": {
            "pdf_dir": str(tmp_path),
            "max_concurrency": 2,
            "max_pages": 20,
            "max_file_size_mb": 20,
            "do_ocr": False,
            "do_table_structure": True,
        }
    }


@pytest.mark.asyncio
async def test_parse_pdf_success(tmp_path, config):
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF- ok")

    service = PDFParserService(config)

    fake_pdf_content = PdfContent(
        sections=[],
        figures=[],
        tables=[],
        raw_text="hello",
        references=[],
        parser_used=ParserType.DOCLING,
        metadata={}
    )

    with patch.object(service.parser, "parse_pdf", AsyncMock(return_value=fake_pdf_content)):
        result = await service.parse_pdf(pdf_file)

    assert isinstance(result, PdfContent)
    assert result.raw_text == "hello"


@pytest.mark.asyncio
async def test_parse_pdf_missing_file(config):
    service = PDFParserService(config)

    with pytest.raises(PDFValidationError):
        await service.parse_pdf(Path("/no/file.pdf"))


@pytest.mark.asyncio
async def test_parse_pdf_raises_unexpected_exception(tmp_path, config):
    pdf_file = tmp_path / "paper.pdf"
    pdf_file.write_bytes(b"%PDF- ok")

    service = PDFParserService(config)

    with patch.object(service.parser, "parse_pdf", AsyncMock(side_effect=RuntimeError("boom"))):
        with pytest.raises(PDFParsingException):
            await service.parse_pdf(pdf_file)
