import pytest

from src.custom.chunker.arxivchunker import TextChunker
from src.custom.extractors.schemas.arxivschema import (
    PdfContent,
    PaperSection,
    ParserType,
)


@pytest.fixture
def chunker():
    return TextChunker(
        chunk_size=200,
        overlap_size=50,
        min_chunk_size=50,
    )


@pytest.fixture
def sample_pdf_with_sections():
    return PdfContent(
        sections=[
            PaperSection(
                title="Introduction",
                content=" ".join(["intro"] * 120),
            ),
            PaperSection(
                title="Method",
                content=" ".join(["method"] * 900),
            ),
        ],
        tables=[],
        figures=[],
        raw_text="",
        references=[],
        parser_used=ParserType.DOCLING,
        metadata={"arxiv_id": "test123"},
    )


@pytest.fixture
def sample_pdf_raw_text_only():
    return PdfContent(
        sections=[],
        tables=[],
        figures=[],
        raw_text=" ".join(["raw"] * 600),
        references=[],
        parser_used=ParserType.DOCLING,
        metadata={"arxiv_id": "raw123"},
    )


def test_section_based_chunking(chunker, sample_pdf_with_sections):
    chunks = chunker.chunk_pdf(sample_pdf_with_sections)

    assert len(chunks) > 0

    # First chunk checks
    first = chunks[0]
    assert first.arxiv_id == "test123"
    assert first.metadata.section_title == "Introduction"
    assert first.metadata.word_count > 0
    assert first.text.strip() != ""


def test_large_section_is_split(chunker, sample_pdf_with_sections):
    chunks = chunker.chunk_pdf(sample_pdf_with_sections)

    method_chunks = [
        c for c in chunks if c.metadata.section_title == "Method"
    ]

    assert len(method_chunks) > 1  # large section must be split


def test_raw_text_fallback(chunker, sample_pdf_raw_text_only):
    chunks = chunker.chunk_pdf(sample_pdf_raw_text_only)

    assert len(chunks) > 0
    assert all(c.metadata.section_title in ("Body", "Full Document") for c in chunks)
    assert chunks[0].arxiv_id == "raw123"


def test_overlap_metadata(chunker, sample_pdf_with_sections):
    chunks = chunker.chunk_pdf(sample_pdf_with_sections)

    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.metadata.overlap_with_previous == 0
        else:
            assert chunk.metadata.overlap_with_previous == chunker.overlap_size


def test_chunk_text_not_empty(chunker, sample_pdf_with_sections):
    chunks = chunker.chunk_pdf(sample_pdf_with_sections)

    for chunk in chunks:
        assert chunk.text.strip() != ""
