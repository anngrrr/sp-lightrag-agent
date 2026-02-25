from nicegui import run, ui

from app.agent.graph import run_agent

MAX_UI_ENTITIES = 10
MAX_UI_RELATIONSHIPS = 10
MAX_UI_CHUNKS = 12


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
                ui.label("Assistant").classes("text-weight-bold")
                answer_markdown = ui.markdown("Thinking...")
                grounding_column = ui.column().classes("w-full gap-2")

        try:
            result = await run.io_bound(run_agent, question)
            answer_markdown.set_content(result.answer or "")
            if result.error is not None:
                ui.notify(
                    f"{result.error.code}: {result.error.message}", type="warning"
                )

            grounding_column.clear()
            with grounding_column:
                ui.separator()
                ui.label("Grounding summary").classes("text-weight-bold")
                ui.label(
                    f"References: {len(result.references)} | "
                    f"Entities: {len(result.entities)} | "
                    f"Relationships: {len(result.relationships)} | "
                    f"Chunks: {len(result.retrieved_chunks)}"
                ).classes("text-caption")

                if result.references:
                    with ui.expansion(
                        "Reference documents", icon="description"
                    ).classes("w-full"):
                        for ref in result.references:
                            ui.label(f"[{ref.reference_id}] {ref.source}")

                if result.entities:
                    with ui.expansion("Key entities", icon="hub").classes("w-full"):
                        for entity in result.entities[:MAX_UI_ENTITIES]:
                            with ui.card().classes("w-full"):
                                ui.label(
                                    f"{entity.entity_name} ({entity.entity_type})"
                                ).classes("text-weight-medium")
                                if entity.description:
                                    ui.label(entity.description)
                                ui.label(entity.source).classes("text-caption")

                if result.relationships:
                    with ui.expansion("Key relationships", icon="account_tree").classes(
                        "w-full"
                    ):
                        for rel in result.relationships[:MAX_UI_RELATIONSHIPS]:
                            with ui.card().classes("w-full"):
                                ui.label(f"{rel.src_id} -> {rel.tgt_id}").classes(
                                    "text-weight-medium"
                                )
                                if rel.description:
                                    ui.label(rel.description)
                                if rel.keywords:
                                    ui.label(f"keywords: {rel.keywords}").classes(
                                        "text-caption"
                                    )
                                if rel.weight is not None:
                                    ui.label(f"weight: {rel.weight:.3f}").classes(
                                        "text-caption"
                                    )
                                ui.label(rel.source).classes("text-caption")

                if result.retrieved_chunks:
                    with ui.expansion("Retrieved chunks", icon="article").classes(
                        "w-full"
                    ):
                        for chunk in result.retrieved_chunks[:MAX_UI_CHUNKS]:
                            with ui.card().classes("w-full"):
                                ui.label(chunk.source).classes("text-weight-medium")
                                ui.label(f"chunk_id: {chunk.chunk_id}").classes(
                                    "text-caption"
                                )
                                ui.label(chunk.content)
        except Exception as exc:
            answer_markdown.set_content("Request failed")
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
