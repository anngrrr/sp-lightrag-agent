import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
from pathlib import Path
from typing import Any, Literal

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc

from app.config import resolve_lightrag_dir
from app.models.schemas import RetrievedChunk

RETRIEVE_MODE = os.getenv("RAG_RETRIEVE_MODE", "mix")
RETRIEVE_TOP_K = int(os.getenv("RAG_RETRIEVE_TOP_K", "8"))
RETRIEVE_CHUNK_TOP_K = int(os.getenv("RAG_RETRIEVE_CHUNK_TOP_K", "8"))
RETRIEVE_TIMEOUT_SEC = float(os.getenv("RAG_RETRIEVE_TIMEOUT_SEC", "20"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text-v1.5")

_RAG_INSTANCE: LightRAG | None = None
_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_VALID_RETRIEVE_MODES = {"local", "global", "hybrid", "naive", "mix", "bypass"}


def _resolve_mode(raw_mode: str) -> Literal[
    "local", "global", "hybrid", "naive", "mix", "bypass"
]:
    if raw_mode in _VALID_RETRIEVE_MODES:
        return raw_mode  # type: ignore[return-value]
    return "mix"


def _read_embedding_dim(storage_dir: Path) -> int:
    vdb_file = storage_dir / "vdb_chunks.json"
    if not vdb_file.exists():
        return 768

    with vdb_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    dim = data.get("embedding_dim")
    if isinstance(dim, int) and dim > 0:
        return dim
    return 768


def _build_embedding_func(storage_dir: Path) -> EmbeddingFunc:
    embedding_dim = _read_embedding_dim(storage_dir)
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        model_name=None,
        func=partial(ollama_embed.func, embed_model=EMBED_MODEL, host=OLLAMA_HOST),
    )


def _get_rag() -> LightRAG:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is not None:
        return _RAG_INSTANCE

    storage_dir = resolve_lightrag_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)

    _RAG_INSTANCE = LightRAG(
        working_dir=str(storage_dir),
        embedding_func=_build_embedding_func(storage_dir),
    )
    return _RAG_INSTANCE


def _query_data(question: str) -> dict[str, Any]:
    rag = _get_rag()
    return rag.query_data(
        question,
        param=QueryParam(
            mode=_resolve_mode(RETRIEVE_MODE),
            top_k=RETRIEVE_TOP_K,
            chunk_top_k=RETRIEVE_CHUNK_TOP_K,
        ),
    )


def retrieve(question: str) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from local LightRAG storage."""
    if not question.strip():
        return []

    future = _EXECUTOR.submit(_query_data, question)
    try:
        result = future.result(timeout=RETRIEVE_TIMEOUT_SEC)
    except FuturesTimeoutError:
        return []
    except Exception:
        return []

    if result.get("status") != "success":
        return []

    data = result.get("data", {})
    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        return []

    retrieved: list[RetrievedChunk] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        chunk_id = str(chunk.get("chunk_id") or chunk.get("reference_id") or "")
        source = str(chunk.get("file_path") or "unknown")
        if not chunk_id:
            continue
        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                source=source,
                content=content,
                score=None,
            )
        )
    return retrieved
