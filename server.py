"""Gradio endpoint and local web-server construction."""

from pathlib import Path
from typing import Optional

import gradio as gr
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from engine import (
    ModelManager,
    ablate_feature_payload,
    analyze_prompt_payload,
    clear_ablation_payload,
    load_model_payload,
)
from inspection_views import inspect_node_payload, options_payload


LOCAL_SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860
SHARE_PUBLICLY = False
SPA_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"


def create_app(manager: Optional[ModelManager] = None) -> gr.Blocks:
    """Build the Gradio app exposing JSON endpoints; never launch here."""
    manager = manager or ModelManager()
    with gr.Blocks(title="Circuit Tracer") as demo:

        def load_endpoint(path: str) -> dict:
            return load_model_payload(manager, path)

        def analyze_endpoint(prompt: str) -> dict:
            return analyze_prompt_payload(manager, prompt)

        def ablate_endpoint(
            node_key: str,
            dims: list[int],
            mode: str,
            scope: str,
            position: Optional[int] = None,
        ) -> dict:
            return ablate_feature_payload(
                manager,
                node_key,
                dims,
                mode,
                scope,
                position,
            )

        def clear_ablation_endpoint() -> dict:
            return clear_ablation_payload(manager)

        def inspect_endpoint(
            node_key: Optional[str] = None,
            token_position: Optional[int] = None,
            view: str = "baseline",
            highlight_token: Optional[str] = None,
            deembed: bool = False,
        ) -> dict:
            return inspect_node_payload(
                manager,
                node_key,
                token_position,
                view,
                highlight_token,
                deembed,
            )

        def get_options(_unused: str = "") -> dict:
            return options_payload()

        gr.api(load_endpoint, api_name="load_checkpoint")
        gr.api(analyze_endpoint, api_name="analyze_prompt")
        gr.api(ablate_endpoint, api_name="ablate_feature")
        gr.api(clear_ablation_endpoint, api_name="clear_ablation")
        gr.api(inspect_endpoint, api_name="inspect_node")
        gr.api(get_options, api_name="options")
    return demo


def fastapi_app(
    manager: Optional[ModelManager] = None,
    static_dir: Optional[Path] = None,
):
    """Wrap the Gradio engine and the frontend build in one FastAPI app."""
    manager = manager or ModelManager()
    demo = create_app(manager)
    demo.queue()

    from gradio.routes import mount_gradio_app

    application = FastAPI()
    application = mount_gradio_app(application, demo, path="/gradio")
    directory = static_dir or SPA_DIST_DIR
    if directory.is_dir():
        application.mount(
            "/", StaticFiles(directory=str(directory), html=True), name="spa"
        )
    return application


def main() -> None:
    import uvicorn

    if not SPA_DIST_DIR.is_dir():
        print(
            "Frontend build not found. Run `npm install && npm run build` "
            f"in the frontend folder, or use `npm run dev` against this server "
            f"({LOCAL_SERVER_NAME}:{SERVER_PORT})."
        )
    uvicorn.run(
        fastapi_app(),
        host=LOCAL_SERVER_NAME,
        port=SERVER_PORT,
        log_level="info",
    )


__all__ = [
    "LOCAL_SERVER_NAME",
    "SERVER_PORT",
    "SHARE_PUBLICLY",
    "SPA_DIST_DIR",
    "create_app",
    "fastapi_app",
    "main",
]
