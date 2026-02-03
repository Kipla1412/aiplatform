import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from src.custom.connectors.ollama import OllamaConnector
from src.custom.llm.exceptions.ollamaexceptions import OllamaConnectionError, OllamaException, OllamaTimeoutError
from src.custom.llm.ragprompt import RAGPromptBuilder, ResponseParser
from src.custom.llm.prompts.promptloader import load_system_prompt

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama local LLM service."""

    def __init__(self, connector: OllamaConnector, config: Dict[str, Any]):
        """
        Initialize Ollama client with:
        - HTTP connector (OllamaConnector)
        - Credentials (from CredentialFactory)
        """
        self.connector = connector

        # # Load credentials once
        # self.creds = CredentialFactory.get_provider("ollama").get_credentials()
        self.default_model = config["default_model"]
        system_prompt = load_system_prompt("rag.yml")
        self.prompt_builder = RAGPromptBuilder(system_prompt)
       
        self.response_parser = ResponseParser()

    async def _get_client(self) -> httpx.AsyncClient:
        """Helper to get active HTTP client from connector."""
        try:
            return await self.connector.connect()
        except httpx.ConnectError as e:
            raise OllamaConnectionError(f"Cannot connect to Ollama service: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama service is healthy and responding.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/version")

            if response.status_code == 200:
                version_data = response.json()
                return {
                    "status": "healthy",
                    "message": "Ollama service is running",
                    "version": version_data.get("version", "unknown"),
                }
            else:
                raise OllamaException(
                    f"Ollama returned status {response.status_code}"
                )

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Ollama health check failed: {str(e)}")

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.
        """
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")

            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            else:
                raise OllamaException(
                    f"Failed to list models: {response.status_code}"
                )

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error listing models: {e}")

    async def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate text using specified model.
        """
        try:
            client = await self._get_client()

            data = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                **kwargs,
            }

            logger.info(
                f"Sending request to Ollama: model={model}, "
                f"stream={stream}, extra_params={kwargs}"
            )

            response = await client.post("/api/generate", json=data)

            if response.status_code != 200:
                raise OllamaException(
                    f"Generation failed: {response.status_code}"
                )

            result = response.json()

            # ---- Usage metadata (same as your reference client) ----
            usage_metadata: Dict[str, Any] = {}

            if "prompt_eval_count" in result:
                usage_metadata["prompt_tokens"] = result.get(
                    "prompt_eval_count", 0
                )
            if "eval_count" in result:
                usage_metadata["completion_tokens"] = result.get(
                    "eval_count", 0
                )

            if usage_metadata:
                usage_metadata["total_tokens"] = (
                    usage_metadata.get("prompt_tokens", 0)
                    + usage_metadata.get("completion_tokens", 0)
                )

            if "total_duration" in result:
                usage_metadata["latency_ms"] = round(
                    result["total_duration"] / 1_000_000, 2
                )

            if "prompt_eval_duration" in result:
                usage_metadata["prompt_eval_duration_ms"] = round(
                    result["prompt_eval_duration"] / 1_000_000, 2
                )
            if "eval_duration" in result:
                usage_metadata["eval_duration_ms"] = round(
                    result["eval_duration"] / 1_000_000, 2
                )

            result["usage_metadata"] = usage_metadata

            logger.debug(f"Usage metadata: {usage_metadata}")

            return result

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error generating with Ollama: {e}")

    async def generate_stream(self, model: str, prompt: str, **kwargs):
        """
        Generate text with streaming response.
        """
        try:
            client = await self._get_client()

            data = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                **kwargs,
            }

            logger.info(f"Starting streaming generation: model={model}")

            async with client.stream("POST", "/api/generate", json=data) as response:
                if response.status_code != 200:
                    raise OllamaException(
                        f"Streaming generation failed: {response.status_code}"
                    )

                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.warning(
                                f"Failed to parse streaming chunk: {line}"
                            )
                            continue

        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(f"Ollama service timeout: {e}")
        except OllamaException:
            raise
        except Exception as e:
            raise OllamaException(f"Error in streaming generation: {e}")

    async def generate_rag_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
        use_structured_output: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate a RAG answer using retrieved chunks.
        """

        try:
            # Use default model from credentials if not provided
            if model is None:
                model = self.default_model 

            if use_structured_output:
                prompt_data = self.prompt_builder.create_structured_prompt(
                    query, chunks
                )

                response = await self.generate(
                    model=model,
                    prompt=prompt_data["prompt"],
                    temperature=0.7,
                    top_p=0.9,
                    format=prompt_data["format"],
                )
            else:
                prompt = self.prompt_builder.create_rag_prompt(query, chunks)

                response = await self.generate(
                    model=model,
                    prompt=prompt,
                    temperature=0.7,
                    top_p=0.9,
                )

            if response and "response" in response:
                answer_text = response["response"]
                logger.debug(
                    f"Raw LLM response: {answer_text[:500]}"
                )

                if use_structured_output:
                    parsed_response = (
                        self.response_parser.parse_structured_response(
                            answer_text
                        )
                    )
                    logger.debug(
                        f"Parsed response: {parsed_response}"
                    )
                    return parsed_response

                # ---- Plain text RAG output (same as reference) ----
                sources = []
                seen_urls = set()

                for chunk in chunks:
                    arxiv_id = chunk.get("arxiv_id")
                    if arxiv_id:
                        arxiv_id_clean = (
                            arxiv_id.split("v")[0]
                            if "v" in arxiv_id
                            else arxiv_id
                        )
                        pdf_url = (
                            f"https://arxiv.org/pdf/{arxiv_id_clean}.pdf"
                        )
                        if pdf_url not in seen_urls:
                            sources.append(pdf_url)
                            seen_urls.add(pdf_url)

                citations = list(
                    set(
                        chunk.get("arxiv_id")
                        for chunk in chunks
                        if chunk.get("arxiv_id")
                    )
                )

                return {
                    "answer": answer_text,
                    "sources": sources,
                    "confidence": "medium",
                    "citations": citations[:5],
                }
            else:
                raise OllamaException(
                    "No response generated from Ollama"
                )

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise OllamaException(
                f"Failed to generate RAG answer: {e}"
            )

    async def generate_rag_answer_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: Optional[str] = None,
    ):
        """
        Streaming RAG answer.
        """

        try:
            if model is None:
                model = self.default_model 


            prompt = self.prompt_builder.create_rag_prompt(
                query, chunks
            )

            async for chunk in self.generate_stream(
                model=model,
                prompt=prompt,
                temperature=0.7,
                top_p=0.9,
            ):
                yield chunk

        except Exception as e:
            logger.error(
                f"Error generating streaming RAG answer: {e}"
            )
            raise OllamaException(
                f"Failed to generate streaming RAG answer: {e}"
            )
