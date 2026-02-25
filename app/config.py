from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIGHTRAG_DIR = PROJECT_ROOT / "data" / "lightrag_storage"


def resolve_lightrag_dir() -> Path:
    """Return LightRAG storage path."""
    return DEFAULT_LIGHTRAG_DIR
