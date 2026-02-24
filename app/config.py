from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIGHTRAG_DIR = PROJECT_ROOT / "data" / "lightrag_storage"
LEGACY_LIGHTRAG_DIR = PROJECT_ROOT / ".storage"


def resolve_lightrag_dir() -> Path:
    """Return existing LightRAG storage path with legacy fallback."""
    if DEFAULT_LIGHTRAG_DIR.exists():
        return DEFAULT_LIGHTRAG_DIR
    if LEGACY_LIGHTRAG_DIR.exists():
        return LEGACY_LIGHTRAG_DIR
    return DEFAULT_LIGHTRAG_DIR
