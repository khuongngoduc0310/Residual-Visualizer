import gc
import html
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import gradio as gr
import tensorflow as tf

from checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from model import ARCHITECTURE_NAME, ModelConfig


LOGGER = logging.getLogger(__name__)
LOCAL_SERVER_NAME = "127.0.0.1"
SHARE_PUBLICLY = False


APP_CSS = """
.ct-page {
    max-width: 1180px;
    margin: 0 auto;
}
.ct-subtitle {
    color: #64748b;
    font-size: 1.05rem;
}
.ct-status {
    border: 1px solid #dbe4ef;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    background: #f8fafc;
}
.ct-diagram {
    display: flex;
    flex-wrap: wrap;
    align-items: stretch;
    gap: 0.65rem;
    padding: 0.4rem 0;
}
.ct-stage {
    flex: 1 1 180px;
    min-width: 160px;
    border: 1px solid #cbd5e1;
    border-radius: 12px;
    padding: 0.85rem;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
}
.ct-stage-label {
    color: #2563eb;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.ct-stage-name {
    color: #0f172a;
    font-weight: 650;
    margin-top: 0.35rem;
}
.ct-stage-detail {
    color: #64748b;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.78rem;
    margin-top: 0.45rem;
}
.ct-arrow {
    align-self: center;
    color: #94a3b8;
    font-size: 1.25rem;
}
@media (max-width: 720px) {
    .ct-arrow {
        display: none;
    }
    .ct-stage {
        flex-basis: 100%;
    }
}
"""


@dataclass(frozen=True)
class ComputeDevice:
    label: str
    tf_device: str
    is_gpu: bool


@dataclass(frozen=True)
class LoadedState:
    checkpoint: LoadedCheckpoint
    checkpoint_path: Path
    device: ComputeDevice


@dataclass(frozen=True)
class LoadResult:
    success: bool
    status: str
    metadata: str
    device: str
    diagram: str
    summary: str


def detect_compute_device() -> ComputeDevice:
    """Return a GPU only when this TensorFlow build and runtime expose one."""
    try:
        build_info = tf.sysconfig.get_build_info()
        is_cuda_build = bool(build_info.get("is_cuda_build", False))
    except (AttributeError, TypeError):
        is_cuda_build = False

    try:
        is_cuda_build = is_cuda_build or bool(tf.test.is_built_with_cuda())
    except (AttributeError, RuntimeError):
        pass

    try:
        physical_gpus = tf.config.list_physical_devices("GPU")
    except (AttributeError, RuntimeError):
        physical_gpus = []

    if not is_cuda_build or not physical_gpus:
        return ComputeDevice(label="CPU", tf_device="/CPU:0", is_gpu=False)

    gpu_name = None
    try:
        details = tf.config.experimental.get_device_details(physical_gpus[0])
        gpu_name = details.get("device_name")
    except (AttributeError, RuntimeError, TypeError):
        pass
    if not gpu_name:
        gpu_name = physical_gpus[0].name
    return ComputeDevice(
        label=f"CUDA GPU: {gpu_name}",
        tf_device="/GPU:0",
        is_gpu=True,
    )


def format_model_summary(model) -> str:
    lines = []
    model.summary(expand_nested=True, print_fn=lines.append)
    return "\n".join(lines)


def _config_details(config: Optional[ModelConfig]) -> dict:
    if config is None:
        return {
            "embedding": "awaiting checkpoint",
            "attention": "awaiting checkpoint",
            "ffn": "awaiting checkpoint",
            "output": "awaiting checkpoint",
        }
    return {
        "embedding": f"sequence <= {config.max_len}, width {config.embedding_dim}",
        "attention": (
            f"{config.num_heads} heads x {config.key_dim}, "
            f"width {config.embedding_dim}"
        ),
        "ffn": f"hidden width {config.feed_forward_dim}",
        "output": f"softmax over {config.vocab_size} tokens",
    }


def render_model_diagram(config: Optional[ModelConfig] = None) -> str:
    """Render stable stage keys for future internal-location highlighting."""
    details = _config_details(config)
    stages = [
        ("embedding", "Residual Stream", "Token + position embeddings", details["embedding"]),
        ("attention_update", "Attention", "Causal attention update", details["attention"]),
        ("attention_residual", "Residual Stream", "Attention residual addition", "input + update"),
        ("attention_norm", "Residual Stream", "First layer normalization", "post-norm"),
        ("ffn_hidden", "FFN", "FFN hidden activation", details["ffn"]),
        ("ffn_update", "FFN", "FFN update", f"width {config.embedding_dim if config else '?'}"),
        ("ffn_residual", "Residual Stream", "FFN residual addition", "normalized attention + update"),
        ("output_norm", "Residual Stream", "Final layer normalization", "post-norm block output"),
        ("vocabulary_projection", "Output", "Vocabulary projection + softmax", details["output"]),
    ]
    cards = []
    for index, (key, category, name, detail) in enumerate(stages):
        cards.append(
            "<div class=\"ct-stage\" data-stage=\"{}\">"
            "<div class=\"ct-stage-label\">{}</div>"
            "<div class=\"ct-stage-name\">{}</div>"
            "<div class=\"ct-stage-detail\">{}</div>"
            "</div>".format(
                html.escape(key),
                html.escape(category),
                html.escape(name),
                html.escape(detail),
            )
        )
        if index < len(stages) - 1:
            cards.append('<div class="ct-arrow" aria-hidden="true">&rarr;</div>')
    return '<div class="ct-diagram">{}</div>'.format("".join(cards))


def _format_metadata(path: Path, checkpoint: LoadedCheckpoint) -> str:
    config = checkpoint.config
    return "\n".join(
        [
            f"**Checkpoint:** `{html.escape(str(path))}`",
            f"**Architecture:** `{html.escape(ARCHITECTURE_NAME)}`",
            f"**Vocabulary:** `{config.vocab_size:,}` tokens",
            f"**Maximum sequence length:** `{config.max_len}` tokens",
            f"**Model width:** `{config.embedding_dim}`",
            f"**Attention:** `{config.num_heads}` heads x `{config.key_dim}` key width",
            f"**FFN width:** `{config.feed_forward_dim}`",
        ]
    )


def _success_result(
    path: Path,
    state: LoadedState,
) -> LoadResult:
    return LoadResult(
        success=True,
        status="Model loaded successfully.",
        metadata=_format_metadata(path, state.checkpoint),
        device=f"**Compute device:** `{html.escape(state.device.label)}`",
        diagram=render_model_diagram(state.checkpoint.config),
        summary=format_model_summary(state.checkpoint.model),
    )


def _failure_result(message: str) -> LoadResult:
    return LoadResult(
        success=False,
        status=f"No model loaded. {message}",
        metadata="",
        device="**Compute device:** `not loaded`",
        diagram=render_model_diagram(),
        summary="",
    )


class ModelManager:
    """Own the single server-side checkpoint used by the application."""

    def __init__(
        self,
        checkpoint_loader: Callable[[str], LoadedCheckpoint] = load_checkpoint,
        device_detector: Callable[[], ComputeDevice] = detect_compute_device,
        session_clearer: Callable[[], None] = tf.keras.backend.clear_session,
        collector: Callable[[], int] = gc.collect,
    ) -> None:
        self._checkpoint_loader = checkpoint_loader
        self._device_detector = device_detector
        self._session_clearer = session_clearer
        self._collector = collector
        self._lock = threading.RLock()
        self._state: Optional[LoadedState] = None

    @property
    def loaded_state(self) -> Optional[LoadedState]:
        with self._lock:
            return self._state

    def _clear_unlocked(self) -> None:
        self._state = None
        self._session_clearer()
        self._collector()

    def clear(self) -> None:
        with self._lock:
            self._clear_unlocked()

    @contextmanager
    def use_loaded_state(self) -> Iterator[LoadedState]:
        """Hold the lifecycle lock while a future analysis uses the model."""
        with self._lock:
            if self._state is None:
                raise CheckpointError("Load a checkpoint before running analysis")
            yield self._state

    def load(self, directory) -> LoadResult:
        with self._lock:
            self._clear_unlocked()
            if not isinstance(directory, str) or not directory.strip():
                return _failure_result("Enter a checkpoint folder path first.")

            checkpoint_path = Path(directory.strip()).expanduser()
            try:
                device = self._device_detector()
                with tf.device(device.tf_device):
                    checkpoint = self._checkpoint_loader(str(checkpoint_path))
                self._state = LoadedState(
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    device=device,
                )
                return _success_result(checkpoint_path, self._state)
            except CheckpointError as error:
                return _failure_result(f"Checkpoint could not be loaded: {error}")
            except (OSError, ValueError, RuntimeError) as error:
                LOGGER.exception("Checkpoint loading failed")
                return _failure_result(
                    f"Checkpoint could not be loaded: {type(error).__name__}. "
                    "Check the folder and runtime configuration."
                )
            except Exception:
                LOGGER.exception("Unexpected checkpoint loading failure")
                return _failure_result(
                    "Checkpoint could not be loaded because of an unexpected runtime error."
                )


def load_model_callback(directory, manager: ModelManager):
    result = manager.load(directory)
    return (
        result.status,
        result.metadata,
        result.device,
        result.diagram,
        result.summary,
    )


def launch_kwargs() -> dict:
    return {
        "server_name": LOCAL_SERVER_NAME,
        "share": SHARE_PUBLICLY,
    }


def create_app(manager: Optional[ModelManager] = None) -> gr.Blocks:
    manager = manager or ModelManager()
    with gr.Blocks(title="Circuit Tracer") as demo:
        with gr.Column(elem_classes="ct-page"):
            gr.Markdown("# Circuit Tracer")
            gr.Markdown(
                "Explore the residual stream of a trained one-block language model.",
                elem_classes="ct-subtitle",
            )
            with gr.Row():
                checkpoint_folder = gr.Textbox(
                    label="Checkpoint folder (server path)",
                    placeholder="/mnt/c/checkpoints/checkpoint-20260902",
                    scale=4,
                )
                load_button = gr.Button("Load Model", variant="primary", scale=1)
            status = gr.Markdown(
                "No model loaded. Enter an extracted checkpoint folder and press Load Model.",
                elem_classes="ct-status",
            )
            with gr.Row():
                metadata = gr.Markdown("", label="Checkpoint details")
                device = gr.Markdown("**Compute device:** `not loaded`", label="Runtime")
            gr.Markdown("## Model architecture")
            diagram = gr.HTML(render_model_diagram())
            with gr.Accordion("Technical model summary", open=False):
                summary = gr.Textbox(
                    value="",
                    lines=20,
                    max_lines=40,
                    label="model.summary()",
                    interactive=False,
                )
            load_button.click(
                fn=lambda directory: load_model_callback(directory, manager),
                inputs=checkpoint_folder,
                outputs=[status, metadata, device, diagram, summary],
                show_progress="minimal",
            )
    return demo


def main() -> None:
    create_app().launch(css=APP_CSS, **launch_kwargs())


if __name__ == "__main__":
    main()
