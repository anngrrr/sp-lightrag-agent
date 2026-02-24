from app.models.schemas import ChatOutput, GraphState


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
