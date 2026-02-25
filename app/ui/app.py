from nicegui import run, ui

from app.agent.graph import run_agent


def build_ui() -> None:
    """Build the MVP chat page."""
    ui.label("SP LightRAG Agent").classes("text-h5")
    ui.label("Q&A for construction standards and rules")

    chat_column = ui.column().classes("w-full gap-4")
    question_input = (
        ui.textarea(
            label="Your question",
            placeholder="Ask about SP/SNiP/GOST requirements...",
        )
        .props("autogrow outlined")
        .classes("w-full")
    )
    send_button = ui.button("Ask")

    async def on_ask(_: object | None = None) -> None:
        question = (question_input.value or "").strip()
        if len(question) < 3:
            ui.notify("Question is too short", type="warning")
            return

        send_button.disable()
        with chat_column:
            with ui.card().classes("w-full"):
                ui.label("User").classes("text-weight-bold")
                ui.label(question)
                answer_label = ui.label("Thinking...")
                citations_column = ui.column().classes("w-full gap-1")

        try:
            result = await run.io_bound(run_agent, question)
            answer_label.set_text(result.answer or "")
            if result.error is not None:
                ui.notify(
                    f"{result.error.code}: {result.error.message}", type="warning"
                )

            citations_column.clear()
            if result.citations:
                with citations_column:
                    ui.separator()
                    ui.label("Sources").classes("text-weight-bold")
                    for idx, citation in enumerate(result.citations, start=1):
                        with ui.card().classes("w-full"):
                            ui.label(f"[{idx}] {citation.source}")
                            ui.label(f"chunk_id: {citation.chunk_id}").classes(
                                "text-caption"
                            )
                            ui.label(citation.quote)
        except Exception as exc:
            answer_label.set_text("Request failed")
            ui.notify(str(exc), type="negative")
        finally:
            send_button.enable()
            question_input.value = ""

    send_button.on_click(on_ask)
    question_input.on("keydown.enter", on_ask)


def run_ui() -> None:
    """Run NiceGUI server."""
    build_ui()
    ui.run(title="SP LightRAG Agent")
