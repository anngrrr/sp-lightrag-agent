from nicegui import ui


def build_ui() -> None:
    """Build the initial MVP page."""
    ui.label("SP LightRAG Agent").classes("text-h5")
    ui.label("MVP skeleton is ready.")


def run_ui() -> None:
    """Run NiceGUI server."""
    build_ui()
    ui.run(title="SP LightRAG Agent")
