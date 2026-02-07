import os
from dataclasses import dataclass
from dotenv import load_dotenv
import json
load_dotenv()

@dataclass
class OllamaConfig:
    base_url: str
    timeout_seconds: int
    models: list[str]
    default_model: str

def ollamaconfig() -> OllamaConfig:
    return OllamaConfig(
        base_url=os.getenv("OLLAMA_HOST", "http://172.21.16.1:11434"),
        timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT", "30")),
        models=os.getenv("OLLAMA_MODELS", '["medgemma-multi:latest"]'), # "MedAIBase/MedGemma1.5:4b"
        default_model=os.getenv("OLLAMA_DEFAULT_MODEL", "medgemma-multi:latest "),
    )
