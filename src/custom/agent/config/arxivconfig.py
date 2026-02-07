from typing import Any, Dict
from pydantic import BaseModel, Field

class GraphConfig(BaseModel):
    """Configuration for the entire graph execution.

    This is the configuration used by AgenticRAGService for controlling
    graph behavior, retrieval settings, and execution parameters.
    """

    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
    model: str   #"medgemma-multi:latest"
    temperature: float = 0.0
    top_k: int = 3
    use_hybrid: bool = True
    enable_tracing: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)