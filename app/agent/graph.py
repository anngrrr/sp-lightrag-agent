import os
from typing import Any, cast

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
from app.rag.retriever import query_native

MAX_CITATIONS = int(os.getenv("RAG_MAX_CITATIONS", "8"))
MAX_CITATION_QUOTE_CHARS = int(os.getenv("RAG_CITATION_QUOTE_CHARS", "260"))

_GRAPH: Any | None = None


class AgentState(TypedDict):
    user_input: ChatInput
    retrieved_chunks: list[RetrievedChunk]
    answer: str | None
    citations: list[Citation]
    error: ErrorInfo | None


def _coerce_state(state: AgentState) -> GraphState:
    return GraphState.model_validate(state)


def _chunks_from_native_result(result: dict[str, Any]) -> list[RetrievedChunk]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return []
    chunks = data.get("chunks", [])
    if not isinstance(chunks, list):
        return []

    out: list[RetrievedChunk] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        content = str(chunk.get("content", "")).strip()
        chunk_id = str(chunk.get("chunk_id") or chunk.get("reference_id") or "").strip()
        source = str(chunk.get("file_path") or "unknown")
        if not content or not chunk_id:
            continue
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                source=source,
                content=content,
                score=None,
            )
        )
    return out


def _citations_from_chunks(chunks: list[RetrievedChunk]) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in chunks[:MAX_CITATIONS]:
        quote = chunk.content.replace("\n", " ").strip()
        quote = quote[:MAX_CITATION_QUOTE_CHARS].rstrip()
        citations.append(
            Citation(
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                quote=quote,
            )
        )
    return citations


def _answer_from_native_result(result: dict[str, Any]) -> str:
    llm_response = result.get("llm_response", {})
    if isinstance(llm_response, dict):
        content = llm_response.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    message = result.get("message")
    if isinstance(message, str) and message.strip():
        return message.strip()
    return "No answer returned by LightRAG."


def native_query_node(state: AgentState) -> dict[str, object]:
    graph_state = _coerce_state(state)
    result = query_native(graph_state.user_input.question)
    status = result.get("status")
    if status != "success":
        message = result.get("message")
        error_message = (
            str(message) if message is not None else "LightRAG query failed."
        )
        return {
            "answer": "LightRAG query failed.",
            "error": ErrorInfo(
                code="lightrag_error",
                message=error_message,
            ).model_dump(),
            "retrieved_chunks": [],
            "citations": [],
        }

    chunks = _chunks_from_native_result(result)
    citations = _citations_from_chunks(chunks)
    answer = _answer_from_native_result(result)
    return {
        "answer": answer,
        "error": None,
        "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
        "citations": [citation.model_dump() for citation in citations],
    }


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
    builder.add_node("native_query", native_query_node)
    builder.add_edge(START, "native_query")
    builder.add_edge("native_query", END)

    _GRAPH = builder.compile()
    return _GRAPH


def run_agent(question: str) -> ChatOutput:
    graph = build_graph()
    init_state = GraphState(user_input=ChatInput(question=question))
    result = graph.invoke(init_state.model_dump())
    final_state = GraphState.model_validate(result)
    return state_to_output(final_state)
