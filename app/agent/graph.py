import os
from typing import Any, cast

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.models.schemas import (
    Citation,
    ChatInput,
    ChatOutput,
    ErrorInfo,
    GraphState,
    RetrievedChunk,
)
from app.rag.retriever import retrieve

load_dotenv()

MAX_CONTEXT_CHARS = int(os.getenv("RAG_CONTEXT_MAX_CHARS", "12000"))
MAX_RESPONSE_CHARS = int(os.getenv("RAG_RESPONSE_MAX_CHARS", "1800"))
MAX_CHUNKS_FOR_CONTEXT = int(os.getenv("RAG_MAX_CHUNKS_FOR_CONTEXT", "8"))
MAX_CITATIONS = int(os.getenv("RAG_MAX_CITATIONS", "5"))
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_LLM_HOST = os.getenv("OLLAMA_LLM_HOST")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL")

_LLM: ChatOllama | None = None
_GRAPH: Any | None = None


class AgentState(TypedDict):
    user_input: ChatInput
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None
    citations: list[Citation]
    error: ErrorInfo | None


def _get_llm() -> ChatOllama:
    global _LLM
    if _LLM is None:
        if not OLLAMA_LLM_MODEL or not OLLAMA_LLM_HOST:
            raise RuntimeError("OLLAMA_LLM_MODEL and OLLAMA_LLM_HOST must be set in .env")
        if not OLLAMA_API_KEY:
            raise RuntimeError("OLLAMA_API_KEY must be set in .env")

        auth_headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"}
        _LLM = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_LLM_HOST,
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


def _coerce_state(state: AgentState) -> GraphState:
    return GraphState.model_validate(state)


def retrieve_node(state: AgentState) -> dict[str, object]:
    graph_state = _coerce_state(state)
    question = graph_state.user_input.question
    chunks = retrieve(question)
    return {"retrieved_chunks": [chunk.model_dump() for chunk in chunks]}


def generate_node(state: AgentState) -> dict[str, object]:
    graph_state = _coerce_state(state)
    context = _build_context(graph_state.retrieved_chunks)
    if not context:
        return {
            "answer": "Not enough relevant context in the knowledge base for a grounded answer.",
            "error": ErrorInfo(
                code="no_context",
                message="No relevant chunks were retrieved for generation.",
            ).model_dump(),
        }

    system_prompt = (
        "You are an assistant for construction standards and rules. "
        "If the user question is outside this domain, refuse to answer and say: "
        "'I can only answer questions about construction standards and rules.' "
        "Answer only from provided context. "
        "If context is insufficient or unrelated to the question, explicitly say so. "
        "Do not invent facts, clause numbers, or documents. "
        "Keep the answer concise."
    )
    human_prompt = (
        f"User question:\n{graph_state.user_input.question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer requirements:\n"
        "1) Short direct answer.\n"
        "2) Only facts from context.\n"
        "3) Say when context is insufficient.\n"
    )

    try:
        llm = _get_llm()
        response = llm.invoke([("system", system_prompt), ("human", human_prompt)])
        answer = _truncate_answer(_response_to_text(response.content).strip())
        return {"answer": answer, "error": None}
    except Exception as exc:
        return {
            "answer": "Failed to generate answer. Check Ollama connectivity and credentials.",
            "error": ErrorInfo(code="llm_error", message=str(exc)).model_dump(),
        }


def citations_node(state: AgentState) -> dict[str, object]:
    graph_state = _coerce_state(state)
    citations: list[Citation] = []
    for chunk in graph_state.retrieved_chunks[:MAX_CITATIONS]:
        quote = chunk.content.strip().replace("\n", " ")
        quote = quote[:220].rstrip()
        citations.append(
            Citation(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                quote=quote,
            )
        )
    return {"citations": [citation.model_dump() for citation in citations]}


def state_to_output(state: GraphState) -> ChatOutput:
    return ChatOutput(
        answer=state.answer or "",
        citations=state.citations,
        error=state.error,
    )


def build_graph() -> Any:
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH

    builder = StateGraph(cast(Any, AgentState))
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("citations", citations_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "citations")
    builder.add_edge("citations", END)

    _GRAPH = builder.compile()
    return _GRAPH


def run_agent(question: str) -> ChatOutput:
    graph = build_graph()
    init_state = GraphState(user_input=ChatInput(question=question))
    result = graph.invoke(init_state.model_dump())
    final_state = GraphState.model_validate(result)
    return state_to_output(final_state)
