import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.custom.extractors.arxivparser import ArxivParser
from src.custom.extractors.exceptions.arxivexceptions import PDFValidationError, PDFParsingException


@pytest.fixture
def config():
    return {
        "max_pages": 20,
        "max_file_size_mb": 30,
        "do_ocr": False,
        "do_table_structure": True
    }


@pytest.mark.asyncio
async def test_parse_pdf_success(tmp_path, config):
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF- dummy")

    parser = ArxivParser(config)

    # Mock pdfium page count
    with patch("pypdfium2.PdfDocument") as mock_pdf:
        mock_pdf.return_value.__len__.return_value = 3

        # Mock Docling convert()
        with patch.object(parser, "_converter") as mock_conv:
            mock_doc = MagicMock()
            mock_doc.texts = []
            mock_doc.export_to_text.return_value = "sample text"

            mock_result = MagicMock()
            mock_result.document = mock_doc
            mock_conv.convert.return_value = mock_result

            result = await parser.parse_pdf(pdf_file)

    assert result is not None
    assert result.raw_text == "sample text"
    assert result.sections == []


@pytest.mark.asyncio
async def test_parse_pdf_invalid_header(tmp_path, config):
    pdf_file = tmp_path / "invalid.pdf"
    pdf_file.write_bytes(b"NOTPDF")

    parser = ArxivParser(config)

    result = await parser.parse_pdf(pdf_file)

    assert result is None


@pytest.mark.asyncio
async def test_parse_pdf_too_large(tmp_path, config):
    pdf_file = tmp_path / "big.pdf"
    pdf_file.write_bytes(b"%PDF-" + b"0" * (40 * 1024 * 1024))

    parser = ArxivParser(config)

    result = await parser.parse_pdf(pdf_file)

    assert result is None



@pytest.mark.asyncio
async def test_docling_failure_raises_parsing_exception(tmp_path, config):
    pdf_file = tmp_path / "ok.pdf"
    pdf_file.write_bytes(b"%PDF- header")

    parser = ArxivParser(config)

    with patch("pypdfium2.PdfDocument") as mock_pdf:
        mock_pdf.return_value.__len__.return_value = 2

        with patch.object(parser, "_converter") as mock_conv:
            mock_conv.convert.side_effect = RuntimeError("docling boom")

            with pytest.raises(PDFParsingException):
                await parser.parse_pdf(pdf_file)
