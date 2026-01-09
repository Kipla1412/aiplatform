import logging
import re
from typing import List, Optional

from src.custom.extractors.schemas.arxivschema import PdfContent
from src.custom.chunker.schemas.arxivchunks import ChunkMetadata, TextChunk
logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits parsed PDF content into smaller text chunks.

    Designed for use in RAG pipelines, vector indexing,
    and embedding generation.

    Operates only on parsed content (PdfContent) and
    does not handle PDF parsing, file I/O, or storage.
    """

    def __init__(
        self,
        chunk_size: int = 600,
        overlap_size: int = 120,
        min_chunk_size: int = 120
    ):
        """
        Initialize chunk configuration.

        Parameters
        ----------
        chunk_size : int
            Target number of words per chunk.
        overlap_size : int
            Overlapping words between consecutive chunks.
        min_chunk_size : int
            Minimum chunk words threshold.
        """
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

        logger.info(
            "Chunker initialized | chunk=%s overlap=%s min=%s",
            chunk_size, overlap_size, min_chunk_size
        )


    def _split_words(self, text: str):
        """
        Split text into a list of word tokens.

        Args:
            text (str): Input text.

        Returns:
            list[str]: Tokenized words.
        """
       
        return re.findall(r"\S+", text)

    def _rebuild(self, words):
        """
        Reconstruct text from a list of word tokens.

        Args:
            words (list[str]): Word tokens.

        Returns:
            str: Reconstructed text.
        """

        return " ".join(words)

    def chunk_pdf(self, pdf: PdfContent) -> List[TextChunk]:
       
        """
        Chunk a parsed PDF document into text segments.

        If structured sections are present, section-aware chunking
        is applied. Otherwise, the entire document text is chunked
        as a continuous block.

        Args:
            pdf (PdfContent): Parsed PDF content.

        Returns:
            List[TextChunk]: List of generated text chunks.
        """

        arxiv_id = pdf.metadata.get("arxiv_id", "unknown")

        if pdf.sections and len(pdf.sections) > 0:
            return self._chunk_by_sections(pdf, arxiv_id)

        logger.warning("No sections found — using raw text chunking")
        return self._chunk_raw_text(pdf.raw_text, arxiv_id)


    def _chunk_by_sections(self, pdf: PdfContent, arxiv_id: str):
        """
        Perform section-aware text chunking.

        Sections are processed according to their length:
        - Small sections are merged with neighboring sections.
        - Medium sections form a single chunk.
        - Large sections are split into multiple overlapping chunks.

        Args:
            pdf (PdfContent): Parsed PDF content.
            arxiv_id (str): Associated arXiv identifier.

        Returns:
            List[TextChunk]: Chunked text segments.
        """
        chunks = []
        buffer = []
        buffer_word_count = 0
        index = 0

        for sec in pdf.sections:
            text = (sec.content or "").strip()
            words = self._split_words(text)
            wc = len(words)

            # small → accumulate
            if wc < 100:
                buffer.append((sec.title, text))
                buffer_word_count += wc
                continue

            # flush buffer if pending
            if buffer:
                combined = "\n\n".join([c for _, c in buffer])
                chunks.append(
                    self._build_chunk(
                        combined,
                        index,
                        arxiv_id,
                        f"Combined Small Sections"
                    )
                )
                index += 1
                buffer = []
                buffer_word_count = 0

            # perfect section
            if 100 <= wc <= 800:
                chunks.append(
                    self._build_chunk(text, index, arxiv_id, sec.title)
                )
                index += 1
                continue

            # large section → split
            split_chunks = self._chunk_raw_text(
                text,
                arxiv_id,
                start_index=index,
                section_title=sec.title
            )
            chunks.extend(split_chunks)
            index += len(split_chunks)

        # flush leftover buffer
        if buffer:
            combined = "\n\n".join([c for _, c in buffer])
            chunks.append(
                self._build_chunk(
                    combined,
                    index,
                    arxiv_id,
                    "Trailing Combined Sections"
                )
            )

        logger.info("Section chunking completed -> %s chunks", len(chunks))
        return chunks

    def _chunk_raw_text(
        self,
        text: str,
        arxiv_id: str,
        start_index: int = 0,
        section_title: Optional[str] = None
    ):
        """
        Chunk continuous text without relying on section boundaries.

        This method is used as a fallback when structured sections
        are unavailable.

        Args:
            text (str): Input text.
            arxiv_id (str): Associated arXiv identifier.
            start_index (int): Starting index for chunk numbering.
            section_title (Optional[str]): Section label for metadata.

        Returns:
            List[TextChunk]: Chunked text segments.
        """

        words = self._split_words(text)

        if len(words) <= self.min_chunk_size:
            return [
                self._build_chunk(
                    text,
                    start_index,
                    arxiv_id,
                    section_title or "Full Document"
                )
            ]

        chunks = []
        pos = 0
        index = start_index

        while pos < len(words):
            end = min(pos + self.chunk_size, len(words))
            part = self._rebuild(words[pos:end])

            chunks.append(
                self._build_chunk(
                    part,
                    index,
                    arxiv_id,
                    section_title or "Body"
                )
            )

            index += 1
            pos += self.chunk_size - self.overlap_size

        return chunks


    def _build_chunk(self, text, idx, arxiv_id, section_title):

        """
        Construct a TextChunk object with metadata.

        Args:
            text (str): Chunk text.
            idx (int): Chunk index.
            arxiv_id (str): Associated arXiv identifier.
            section_title (str): Section title for metadata.

        Returns:
            TextChunk: Structured chunk with metadata.
        """
        words = text.split()

        start_char = 0
        end_char = len(text)

        overlap_prev = self.overlap_size if idx > 0 else 0
        overlap_next = self.overlap_size
        return TextChunk(
            text=text,
            arxiv_id=arxiv_id,
            metadata=ChunkMetadata(
                chunk_index=idx,
                section_title=section_title,
                word_count=len(words),
                start_char=start_char,
                end_char=end_char,
                overlap_with_previous=overlap_prev,
                overlap_with_next=overlap_next,
            ),
        )
