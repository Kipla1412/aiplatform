import logging
from pathlib import Path

from src.custom.extractors.pdfparserservice import PDFParserService
from src.custom.credentials.localsettings.parser import ArxivParserConfig
import asyncio
from src.custom.chunker.arxivchunker import TextChunker

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')




def test_realtime_pdf_chunking():
    """
    End-to-end test:
    PDF -> Parser -> PdfContent -> Chunker
    """

    config = ArxivParserConfig()
    parser = PDFParserService({"docling": config.model_dump()})
    chunker = TextChunker(
        chunk_size=500,
        overlap_size=100,
        min_chunk_size=120,
    )

    async def run():
        parsed_docs = await parser.parse_all_pdfs()
        assert len(parsed_docs) > 0, "No PDFs parsed"

        pdf = parsed_docs[0]
        chunks = chunker.chunk_pdf(pdf)

        # ---- Assertions ----
        assert len(chunks) > 0, "No chunks created"
        assert chunks[0].text.strip() != ""
        assert chunks[0].arxiv_id == pdf.metadata["arxiv_id"]

        print(f"PDF: {pdf.metadata['arxiv_id']}")
        print(f"Chunks created: {len(chunks)}")

        for i, chunk in enumerate(chunks[:3], start=1):
            print(f"--- Chunk {i} ---")
            print(f"Section: {chunk.metadata.section_title}")
            print(f"Words: {chunk.metadata.word_count}")
            print(chunk.text[:800])   # preview only
            print("-" * 60)

    asyncio.run(run())

if __name__ == "__main__":
    test_realtime_pdf_chunking()
