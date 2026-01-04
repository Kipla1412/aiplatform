from pathlib import Path
import yaml

def load_yml(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.absolute()}")

    if path.suffix not in ['.yml', '.yaml']:
        raise ValueError(f"File {path.name} is not a valid YAML file.")

    try:
        with path.open('r') as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path.name}: {e}")


        