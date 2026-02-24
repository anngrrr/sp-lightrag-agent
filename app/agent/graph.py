import os

from langchain_ollama import ChatOllama
from dotenv import load_dotenv

from app.models.schemas import ChatOutput, ErrorInfo, GraphState, RetrievedChunk

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
MAX_CONTEXT_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "12000"))
MAX_RESPONSE_CHARS = int(os.getenv("RAG_RESPONSE_MAX_CHARS", "1800"))
MAX_CHUNKS_FOR_CONTEXT = int(os.getenv("RAG_MAX_CHUNKS_FOR_CONTEXT", "8"))

_LLM: ChatOllama | None = None


def _get_llm() -> ChatOllama:
    global _LLM
    if _LLM is None:
        if not OLLAMA_MODEL or not OLLAMA_HOST:
            raise RuntimeError(
                "OLLAMA_MODEL and OLLAMA_HOST must be set in .env"
            )
        if not OLLAMA_API_KEY:
            raise RuntimeError("OLLAMA_API_KEY must be set in .env")

        auth_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        _LLM = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_HOST,
            temperature=0.1,
            num_predict=450,
            client_kwargs={"headers": auth_headers},
            async_client_kwargs={"headers": auth_headers},
        )
    return _LLM


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    context_parts: list[str] = []
    total_chars = 0
    for idx, chunk in enumerate(chunks[:MAX_CHUNKS_FOR_CONTEXT], start=1):
        part = (
            f"[{idx}] source={chunk.source}; chunk_id={chunk.chunk_id}\n"
            f"{chunk.content.strip()}"
        )
        if total_chars + len(part) > MAX_CONTEXT_CHARS:
            break
        context_parts.append(part)
        total_chars += len(part)
    return "\n\n".join(context_parts)


def _truncate_answer(text: str) -> str:
    if len(text) <= MAX_RESPONSE_CHARS:
        return text
    return text[: MAX_RESPONSE_CHARS - 3].rstrip() + "..."


def _response_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item for item in content if isinstance(item, str)]
        return "\n".join(parts)
    return ""


def generate_node(state: GraphState) -> GraphState:
    """Generate answer from retrieved chunks via Ollama."""
    context = _build_context(state.retrieved_chunks)
    if not context:
        state.answer = "Недостаточно релевантного контекста в базе знаний, чтобы дать точный ответ."
        state.error = ErrorInfo(
            code="no_context",
            message="No relevant chunks were retrieved for generation.",
        )
        return state

    system_prompt = (
        "Ты помощник по строительным нормам и правилам. "
        "Отвечай только на основе переданного контекста. "
        "Если в контексте нет точного основания, прямо скажи, что данных недостаточно. "
        "Не выдумывай факты, номера пунктов или нормативные документы. "
        "Пиши коротко и по делу."
    )
    human_prompt = (
        f"Вопрос пользователя:\n{state.user_input.question}\n\n"
        f"Контекст:\n{context}\n\n"
        "Требования к ответу:\n"
        "1) Краткий прямой ответ.\n"
        "2) Только факты из контекста.\n"
        "3) Если данных не хватает - так и напиши.\n"
    )

    try:
        llm = _get_llm()
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        state.answer = _truncate_answer(_response_to_text(response.content).strip())
        state.error = None
    except Exception as exc:
        state.answer = "Не удалось сгенерировать ответ. Проверьте доступность Ollama."
        state.error = ErrorInfo(code="llm_error", message=str(exc))

    return state


def state_to_output(state: GraphState) -> ChatOutput:
    """Map graph state to UI response contract."""
    return ChatOutput(
        answer=state.answer or "",
        citations=state.citations,
        error=state.error,
    )


def build_graph() -> None:
    """Placeholder for LangGraph compilation."""
    return None
