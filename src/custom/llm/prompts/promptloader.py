import yaml
from pathlib import Path


def load_system_prompt(filename: str = "rag.yml") -> str:
    """
    Load RAG system prompt from YAML file.
    """
    base_dir = Path(__file__).parent
    path = base_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"RAG prompt file not found: {path}")

    data = yaml.safe_load(path.read_text())

    try:
        return data["rag"]["system_prompt"]
    except KeyError as e:
        raise KeyError(
            f"Invalid RAG prompt YAML structure, missing key: {e}"
        )