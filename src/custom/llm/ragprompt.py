"""
RAG prompt construction and response parsing utilities.

This module contains:
- RAGPromptBuilder: Builds prompts for Retrieval-Augmented Generation
- ResponseParser: Safely parses and validates LLM responses

"""

import json
import re
from typing import Any, Dict, List

from pydantic import ValidationError
from src.custom.llm.schemas.ollamaschema import RAGResponse


class RAGPromptBuilder:
    """
    Builds prompts for Retrieval-Augmented Generation (RAG).

    Responsibilities:
    - Combine system instructions, retrieved context, and user query
    - Produce a single prompt string suitable for LLM consumption

    This class does NOT:
    - Load files
    - Read environment variables
    - Call LLMs
    """

    def __init__(self, system_prompt: str):
        """
        Initialize the prompt builder.

        Args:
            system_prompt (str): System-level instructions for the LLM.
        """
        self.system_prompt = system_prompt

    def create_rag_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Create a RAG prompt using retrieved document chunks.

        Args:
            query (str): The user's question.
            chunks (List[Dict[str, Any]]): Retrieved document chunks,
                typically returned from OpenSearch.

        Returns:
            str: A fully formatted RAG prompt string.
        """
        prompt = f"{self.system_prompt}\n\n"
        prompt += "### Context from Papers:\n\n"

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk.get("chunk_text", chunk.get("content", ""))
            arxiv_id = chunk.get("arxiv_id", "")

            prompt += f"[{i}. arXiv:{arxiv_id}]\n"
            prompt += f"{chunk_text}\n\n"

        prompt += f"### Question:\n{query}\n\n"
        prompt += (
            "### Answer:\n"
            "Provide a natural, factual answer and cite sources using [arXiv:id].\n\n"
        )

        return prompt

    def create_structured_prompt(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create a RAG prompt with a structured output schema.

        This is used when the LLM supports schema-guided generation
        (e.g., Ollama structured output).

        Args:
            query (str): The user's question.
            chunks (List[Dict[str, Any]]): Retrieved document chunks.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - "prompt": Prompt string
                - "format": JSON schema for structured output
        """
        return {
            "prompt": self.create_rag_prompt(query, chunks),
            "format": RAGResponse.model_json_schema(),
        }


class ResponseParser:
    """
    Parses and validates LLM responses.

    Responsibilities:
    - Convert raw LLM output into structured Python dictionaries
    - Validate responses against Pydantic models
    - Gracefully handle malformed or partial outputs
    """

    @staticmethod
    def parse_structured_response(response: str) -> Dict[str, Any]:
        """
        Parse a structured LLM response.

        Attempts:
        1. Direct JSON parsing
        2. Pydantic validation
        3. Fallback extraction if malformed

        Args:
            response (str): Raw LLM response text.

        Returns:
            Dict[str, Any]: Parsed and validated response dictionary.
        """
        try:
            parsed_json = json.loads(response)
            validated = RAGResponse(**parsed_json)
            return validated.model_dump()
        except (json.JSONDecodeError, ValidationError):
            return ResponseParser._extract_json_fallback(response)

    @staticmethod
    def _extract_json_fallback(response: str) -> Dict[str, Any]:
        """
        Attempt to extract JSON content from a malformed LLM response.

        This is a best-effort fallback and ensures the application
        never crashes due to unexpected LLM output.

        Args:
            response (str): Raw LLM response text.

        Returns:
            Dict[str, Any]: Extracted response or a safe fallback structure.
        """
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                validated = RAGResponse(**parsed)
                return validated.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass

        # Final fallback: return plain text answer
        return {
            "answer": response,
            "sources": [],
            "confidence": "low",
            "citations": [],
        }
