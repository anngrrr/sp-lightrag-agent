from typing import Any, cast

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from app.models.schemas import (
    ChatInput,
    ChatOutput,
    ErrorInfo,
    GraphState,
    RetrievedChunk,
    RetrievedEntity,
    RetrievedReference,
    RetrievedRelationship,
)
from app.rag.retriever import query_native

_GRAPH: Any | None = None


class AgentState(TypedDict):
    user_input: ChatInput
    retrieved_chunks: list[RetrievedChunk]
    entities: list[RetrievedEntity]
    relationships: list[RetrievedRelationship]
    references: list[RetrievedReference]
    answer: str | None
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


def _entities_from_native_result(result: dict[str, Any]) -> list[RetrievedEntity]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return []
    entities = data.get("entities", [])
    if not isinstance(entities, list):
        return []

    out: list[RetrievedEntity] = []
    for item in entities:
        if not isinstance(item, dict):
            continue
        entity_name = str(item.get("entity_name") or "").strip()
        if not entity_name:
            continue
        out.append(
            RetrievedEntity(
                entity_name=entity_name,
                entity_type=str(item.get("entity_type") or "unknown"),
                description=str(item.get("description") or "").strip(),
                source=str(item.get("file_path") or "unknown"),
            )
        )
    return out


def _relationships_from_native_result(
    result: dict[str, Any],
) -> list[RetrievedRelationship]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return []
    relationships = data.get("relationships", [])
    if not isinstance(relationships, list):
        return []

    out: list[RetrievedRelationship] = []
    for item in relationships:
        if not isinstance(item, dict):
            continue
        src_id = str(item.get("src_id") or "").strip()
        tgt_id = str(item.get("tgt_id") or "").strip()
        if not src_id or not tgt_id:
            continue
        weight_raw = item.get("weight")
        weight: float | None = None
        if isinstance(weight_raw, int | float):
            weight = float(weight_raw)
        out.append(
            RetrievedRelationship(
                src_id=src_id,
                tgt_id=tgt_id,
                description=str(item.get("description") or "").strip(),
                keywords=str(item.get("keywords") or "").strip(),
                weight=weight,
                source=str(item.get("file_path") or "unknown"),
            )
        )
    return out


def _references_from_native_result(result: dict[str, Any]) -> list[RetrievedReference]:
    data = result.get("data", {})
    if not isinstance(data, dict):
        return []
    references = data.get("references", [])
    if not isinstance(references, list):
        return []

    out: list[RetrievedReference] = []
    for item in references:
        if not isinstance(item, dict):
            continue
        ref_id = str(item.get("reference_id") or "").strip()
        if not ref_id:
            continue
        out.append(
            RetrievedReference(
                reference_id=ref_id,
                source=str(item.get("file_path") or "unknown"),
            )
        )
    return out


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
            "entities": [],
            "relationships": [],
            "references": [],
        }

    chunks = _chunks_from_native_result(result)
    entities = _entities_from_native_result(result)
    relationships = _relationships_from_native_result(result)
    references = _references_from_native_result(result)
    answer = _answer_from_native_result(result)
    return {
        "answer": answer,
        "error": None,
        "retrieved_chunks": [chunk.model_dump() for chunk in chunks],
        "entities": [entity.model_dump() for entity in entities],
        "relationships": [relation.model_dump() for relation in relationships],
        "references": [reference.model_dump() for reference in references],
    }


def state_to_output(state: GraphState) -> ChatOutput:
    return ChatOutput(
        answer=state.answer or "",
        retrieved_chunks=state.retrieved_chunks,
        entities=state.entities,
        relationships=state.relationships,
        references=state.references,
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
