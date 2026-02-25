import json
import logging
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import partial
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc

from app.config import resolve_lightrag_dir

load_dotenv()

LOGGER = logging.getLogger(__name__)
RETRIEVE_MODE = os.getenv("RAG_RETRIEVE_MODE", "hybrid")
RETRIEVE_TOP_K = int(os.getenv("RAG_RETRIEVE_TOP_K", "24"))
RETRIEVE_CHUNK_TOP_K = int(os.getenv("RAG_RETRIEVE_CHUNK_TOP_K", "12"))
RETRIEVE_TIMEOUT_SEC = float(os.getenv("RAG_RETRIEVE_TIMEOUT_SEC", "20"))
RAG_RESPONSE_TYPE = os.getenv("RAG_RESPONSE_TYPE", "Multiple Paragraphs")
RAG_INCLUDE_REFERENCES = os.getenv("RAG_INCLUDE_REFERENCES", "true").lower() == "true"
RAG_MAX_ENTITY_TOKENS = int(os.getenv("RAG_MAX_ENTITY_TOKENS", "8000"))
RAG_MAX_RELATION_TOKENS = int(os.getenv("RAG_MAX_RELATION_TOKENS", "8000"))
RAG_MAX_TOTAL_TOKENS = int(os.getenv("RAG_MAX_TOTAL_TOKENS", "16000"))
RAG_KEYWORD_LLM_TIMEOUT_SEC = float(os.getenv("RAG_KEYWORD_LLM_TIMEOUT_SEC", "120"))
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_LLM_HOST = os.getenv("OLLAMA_LLM_HOST")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL")
OLLAMA_EMBED_HOST = os.getenv("OLLAMA_EMBED_HOST")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL")

_RAG_INSTANCE: LightRAG | None = None
_EXECUTOR = ThreadPoolExecutor(max_workers=1)
_VALID_RETRIEVE_MODES = {"local", "global", "hybrid", "naive", "mix", "bypass"}


def _resolve_mode(
    raw_mode: str,
) -> Literal["local", "global", "hybrid", "naive", "mix", "bypass"]:
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
    embed_kwargs: dict[str, str] = {}
    if OLLAMA_EMBED_MODEL:
        embed_kwargs["embed_model"] = OLLAMA_EMBED_MODEL
    if OLLAMA_EMBED_HOST:
        embed_kwargs["host"] = OLLAMA_EMBED_HOST
    if OLLAMA_API_KEY:
        embed_kwargs["api_key"] = OLLAMA_API_KEY
    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        model_name=None,
        func=partial(ollama_embed.func, **embed_kwargs),
    )


def _get_rag() -> LightRAG:
    global _RAG_INSTANCE
    if _RAG_INSTANCE is not None:
        return _RAG_INSTANCE

    storage_dir = resolve_lightrag_dir()
    storage_dir.mkdir(parents=True, exist_ok=True)
    if not OLLAMA_LLM_MODEL:
        raise RuntimeError("OLLAMA_LLM_MODEL must be set in .env")

    llm_kwargs: dict[str, Any] = {"host": OLLAMA_LLM_HOST} if OLLAMA_LLM_HOST else {}
    if OLLAMA_API_KEY:
        llm_kwargs["api_key"] = OLLAMA_API_KEY
    llm_kwargs["timeout"] = RAG_KEYWORD_LLM_TIMEOUT_SEC
    llm_kwargs["options"] = {"temperature": 0}
    _RAG_INSTANCE = LightRAG(
        working_dir=str(storage_dir),
        llm_model_func=ollama_model_complete,
        llm_model_name=OLLAMA_LLM_MODEL,
        llm_model_kwargs=llm_kwargs,
        embedding_func=_build_embedding_func(storage_dir),
    )
    asyncio.run(_RAG_INSTANCE.initialize_storages())
    return _RAG_INSTANCE


def _build_query_param() -> QueryParam:
    return QueryParam(
        mode=_resolve_mode(RETRIEVE_MODE),
        top_k=RETRIEVE_TOP_K,
        chunk_top_k=RETRIEVE_CHUNK_TOP_K,
        enable_rerank=False,
        max_entity_tokens=RAG_MAX_ENTITY_TOKENS,
        max_relation_tokens=RAG_MAX_RELATION_TOKENS,
        max_total_tokens=RAG_MAX_TOTAL_TOKENS,
        response_type=RAG_RESPONSE_TYPE,
        include_references=RAG_INCLUDE_REFERENCES,
    )


def _query_llm(question: str) -> dict[str, Any]:
    rag = _get_rag()
    return rag.query_llm(question, param=_build_query_param())


def query_native(question: str) -> dict[str, Any]:
    """Run native LightRAG retrieval + generation pipeline."""
    if not question.strip():
        return {
            "status": "failure",
            "message": "Question is empty",
            "data": {},
            "metadata": {},
            "llm_response": {
                "content": "",
                "response_iterator": None,
                "is_streaming": False,
            },
        }

    future = _EXECUTOR.submit(_query_llm, question)
    try:
        return future.result(timeout=RETRIEVE_TIMEOUT_SEC)
    except FuturesTimeoutError:
        LOGGER.warning(
            "LightRAG native query timed out after %.1f sec", RETRIEVE_TIMEOUT_SEC
        )
        return {
            "status": "failure",
            "message": "LightRAG query timeout",
            "data": {},
            "metadata": {},
            "llm_response": {
                "content": "",
                "response_iterator": None,
                "is_streaming": False,
            },
        }
    except Exception as exc:
        LOGGER.exception("LightRAG native query failed: %s", exc)
        return {
            "status": "failure",
            "message": str(exc),
            "data": {},
            "metadata": {},
            "llm_response": {
                "content": "",
                "response_iterator": None,
                "is_streaming": False,
            },
        }
