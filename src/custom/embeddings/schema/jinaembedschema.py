from typing import Dict, List

from pydantic import BaseModel, Field


class JinaEmbeddingRequest(BaseModel):
    """Request model for Jina embeddings API."""

    model: str = Field(..., description="Jina embedding model name")
    task: str = Field(
        default="retrieval.passage",
        description="retrieval.passage or retrieval.query",
    )
    dimensions: int = Field(..., description="Embedding vector dimensions")
    late_chunking: bool = Field(
        default=False, description="Enable late chunking"
    )
    embedding_type: str = Field(
        default="float", description="Embedding value type"
    )
    input: List[str] = Field(
        ..., description="List of input texts to embed"
    )



class JinaEmbeddingResponse(BaseModel):
    """Response model from Jina embeddings API."""

    model: str
    object: str = "list"
    usage: Dict[str, int]
    data: List[Dict]