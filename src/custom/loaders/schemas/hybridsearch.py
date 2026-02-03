
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class HybridIndexingConfig:
    """Configuration for Hybrid Indexing """

    embedding_model: str = "jina-embeddings-v3"
    embedding_batch_size: int = 50
    text_field: str = "chunk_text"
    replace_existing: bool = False
