import gc
import html
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import tensorflow as tf

import gradio as gr

from analysis import AnalysisError, PromptAnalysis, analyze_prompt
from charts import (
    display_bounds,
    render_activation_heatmap,
    render_token_distribution,
    render_token_magnitudes,
    tensor_statistics,
)
from checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from inspection import (
    DEFAULT_LOCATION_KEY,
    location_choices,
    location_spec,
)
from model import ARCHITECTURE_NAME, ModelConfig


LOGGER = logging.getLogger(__name__)
LOCAL_SERVER_NAME = "127.0.0.1"
SHARE_PUBLICLY = False
LIGHT_THEME = gr.themes.Soft()


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
.ct-two-up,
.ct-chart-row {
    min-width: 0;
}
.ct-two-up > *,
.ct-chart-row > * {
    min-width: 0;
}
.ct-table {
    min-width: 0;
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
.ct-stage.ct-selected {
    border-color: #2563eb;
    background: #eff6ff;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.25);
}
.ct-arrow {
    align-self: center;
    color: #94a3b8;
    font-size: 1.25rem;
}
@media (max-width: 720px) {
    .ct-two-up,
    .ct-chart-row {
        flex-direction: column;
    }
    .ct-table {
        overflow-x: auto;
    }
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
class InspectionSession:
    analysis: PromptAnalysis
    config: ModelConfig


@dataclass(frozen=True)
class LoadResult:
    success: bool
    status: str
    metadata: str
    device: str
    diagram: str
    summary: str


@dataclass(frozen=True)
class AnalysisResult:
    success: bool
    status: str
    token_count: str
    token_rows: list
    unknown_warning: str
    next_token_rows: list


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


def render_model_diagram(
    config: Optional[ModelConfig] = None,
    selected_key: Optional[str] = None,
) -> str:
    """Render the model stages, highlighting the selected internal location."""
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
        classes = "ct-stage ct-selected" if key == selected_key else "ct-stage"
        cards.append(
            "<div class=\"{}\" data-stage=\"{}\">"
            "<div class=\"ct-stage-label\">{}</div>"
            "<div class=\"ct-stage-name\">{}</div>"
            "<div class=\"ct-stage-detail\">{}</div>"
            "</div>".format(
                classes,
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


def _analysis_success_result(analysis: PromptAnalysis) -> AnalysisResult:
    unknown_warning = (
        f"**Warning:** this prompt contains `{analysis.unknown_count}` "
        "unknown token(s), mapped to `[UNK]`."
        if analysis.unknown_count
        else ""
    )
    return AnalysisResult(
        success=True,
        status=f"Analysis complete for `{analysis.token_count}` processed token(s).",
        token_count=(
            f"**Processed tokens:** `{analysis.token_count}` of "
            f"`{analysis.max_len}`"
        ),
        token_rows=[
            [token.position, token.text, token.token_id]
            for token in analysis.tokens
        ],
        unknown_warning=unknown_warning,
        next_token_rows=[
            [token.rank, token.text, token.token_id, f"{token.probability:.4f}"]
            for token in analysis.next_tokens
        ],
    )


def _analysis_failure_result(message: str) -> AnalysisResult:
    return AnalysisResult(
        success=False,
        status=message,
        token_count="",
        token_rows=[],
        unknown_warning="",
        next_token_rows=[],
    )


def _token_choices(analysis: PromptAnalysis) -> list:
    return [
        (f"{token.position}: {token.text}", str(token.position))
        for token in analysis.tokens
    ]


def location_explanation(spec) -> str:
    """Markdown heading, category and plain-language explanation."""
    return (
        f"### {spec.category} \u00b7 {spec.label}\n\n{spec.explanation}"
    )


def location_stats(spec, values: np.ndarray, token_position: int) -> str:
    """Markdown lines describing the captured tensor and selected token."""
    seq_len, width = values.shape
    stats = tensor_statistics(values[token_position])
    return (
        f"**Shape:** `{seq_len} \u00d7 {width}`\n"
        f"**Captured range:** min `{values.min():.4f}`, mean `{values.mean():.4f}`, "
        f"max `{values.max():.4f}`\n"
        f"**Selected token:** norm `{stats.norm:.4f}`, mean `{stats.mean:.4f}`, "
        f"std `{stats.standard_deviation:.4f}`, min `{stats.minimum:.4f}`, "
        f"max `{stats.maximum:.4f}`"
    )


def heatmap_range(values: np.ndarray, clipped: bool) -> str:
    lower, upper = display_bounds(values, clipped)
    suffix = " (display clipped to the 1st-99th percentile)" if clipped else ""
    return f"**Visible heatmap range:** `{lower:.4f}` to `{upper:.4f}`{suffix}"


INSPECT_AWAITING = (
    "Analyze a prompt to capture every internal location."
)

CLICK_BRIDGE_JS = """
() => {
  const install = () => {
    const host = document.querySelector('#ct-magnitude-plot');
    if (!host || host.dataset.clickBridgeInstalled === 'true') return;
    host.dataset.clickBridgeInstalled = 'true';
    host.addEventListener('plotly_click', (event) => {
      const point = event.detail?.points?.[0];
      const position = point?.customdata ?? point?.pointIndex;
      const input = document.querySelector(
        '#ct-token-click input, #ct-token-click textarea'
      );
      if (position === undefined || !input) return;
      const setter = Object.getOwnPropertyDescriptor(
        input instanceof HTMLTextAreaElement
          ? HTMLTextAreaElement.prototype
          : HTMLInputElement.prototype,
        'value'
      ).set;
      setter.call(input, String(position));
      input.dispatchEvent(new Event('input', {bubbles: true}));
      input.dispatchEvent(new Event('change', {bubbles: true}));
    });
  };
  install();
  new MutationObserver(install).observe(document.body, {
    childList: true,
    subtree: true,
  });
}
"""


def _empty_inspection_outputs():
    empty_dropdown = gr.update(choices=[], value=None)
    return (
        empty_dropdown,
        empty_dropdown,
        INSPECT_AWAITING,
        "",
        "",
        None,
        None,
        None,
        render_model_diagram(),
    )


def _successful_inspection_outputs(
    session: InspectionSession,
    location_key: str = DEFAULT_LOCATION_KEY,
    token_position: Optional[int] = None,
    clipped: bool = False,
):
    analysis = session.analysis
    spec = location_spec(location_key)
    token_position = (
        analysis.token_count - 1
        if token_position is None
        else max(0, min(token_position, analysis.token_count - 1))
    )
    location_dropdown = gr.update(choices=location_choices(), value=spec.key)
    token_dropdown = gr.update(
        choices=_token_choices(analysis),
        value=str(token_position),
    )
    values = analysis.capture.locations[spec.key]
    labels = [f"{token.position}: {token.text}" for token in analysis.tokens]
    return (
        location_dropdown,
        token_dropdown,
        location_explanation(spec),
        location_stats(spec, values, token_position),
        heatmap_range(values, clipped),
        render_token_magnitudes(values, labels, token_position),
        render_activation_heatmap(values, labels, token_position, clipped),
        render_token_distribution(
            values[token_position], token_position, labels[token_position]
        ),
        render_model_diagram(session.config, selected_key=spec.key),
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
        self._session: Optional[InspectionSession] = None

    @property
    def loaded_state(self) -> Optional[LoadedState]:
        with self._lock:
            return self._state

    @property
    def inspection_session(self) -> Optional[InspectionSession]:
        """Return the last capture; None when nothing has been analyzed."""
        with self._lock:
            return self._session

    def store_inspection_session(self, session: InspectionSession) -> None:
        with self._lock:
            self._session = session

    def clear_session(self) -> None:
        """Drop the stored capture without unloading the current model."""
        with self._lock:
            self._session = None

    def _clear_unlocked(self) -> None:
        self._state = None
        self._session = None
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


def analyze_prompt_callback(prompt, manager: ModelManager):
    try:
        with manager.use_loaded_state() as state:
            with tf.device(state.device.tf_device):
                result = analyze_prompt(prompt or "", state.checkpoint)
        manager.store_inspection_session(
            InspectionSession(
                analysis=result,
                config=state.checkpoint.config,
            )
        )
        result = _analysis_success_result(result)
    except (AnalysisError, CheckpointError) as error:
        manager.clear_session()
        result = _analysis_failure_result(str(error))
    except Exception:
        manager.clear_session()
        LOGGER.exception("Prompt analysis failed")
        result = _analysis_failure_result(
            "Prompt analysis failed because of an unexpected runtime error."
        )
    return (
        result.status,
        result.token_count,
        result.unknown_warning,
        result.token_rows,
        result.next_token_rows,
    )


def analyze_and_inspect_callback(prompt, manager: ModelManager):
    """Run analysis and refresh the inspect panel from the same single run."""
    core = analyze_prompt_callback(prompt, manager)
    session = manager.inspection_session
    if session is None:
        return (*core, *_empty_inspection_outputs())
    return (*core, *_successful_inspection_outputs(session))


def select_location_callback(
    location_key,
    token_value,
    clipped,
    manager: ModelManager,
):
    """Re-render the inspect panel from stored data; never reruns the model."""
    session = manager.inspection_session
    if session is None:
        return (
            gr.update(value=None),
            INSPECT_AWAITING,
            "",
            "",
            None,
            None,
            None,
            render_model_diagram(),
        )

    analysis = session.analysis
    position = (
        int(token_value)
        if token_value is not None and token_value != ""
        else analysis.token_count - 1
    )
    return _successful_inspection_outputs(
        session,
        location_key or DEFAULT_LOCATION_KEY,
        position,
        bool(clipped),
    )[1:]


def select_clicked_token_callback(
    clicked_position,
    location_key,
    token_value,
    clipped,
    manager: ModelManager,
):
    """Apply a Plotly bar click, then render all views from captured data."""
    session = manager.inspection_session
    if session is None:
        return (
            gr.update(value=None),
            INSPECT_AWAITING,
            "",
            "",
            None,
            None,
            None,
            render_model_diagram(),
        )
    try:
        position = int(clicked_position)
    except (TypeError, ValueError):
        position = int(token_value) if token_value not in (None, "") else None
    outputs = _successful_inspection_outputs(
        session,
        location_key or DEFAULT_LOCATION_KEY,
        position,
        bool(clipped),
    )
    return (
        outputs[1],
        outputs[2],
        outputs[3],
        outputs[4],
        outputs[5],
        outputs[6],
        outputs[7],
        outputs[8],
    )


def launch_kwargs() -> dict:
    return {
        "server_name": LOCAL_SERVER_NAME,
        "share": SHARE_PUBLICLY,
        "theme": LIGHT_THEME,
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
            with gr.Row(elem_classes="ct-two-up"):
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
            gr.Markdown("## Analyze a prompt")
            prompt_text = gr.Textbox(
                label="Prompt",
                lines=3,
                placeholder="Enter text to analyze with the loaded model.",
            )
            analyze_button = gr.Button("Analyze Prompt", variant="primary")
            analysis_status = gr.Markdown(
                "Load a model, enter a prompt, and press Analyze Prompt.",
                elem_classes="ct-status",
            )
            token_count_line = gr.Markdown("")
            unknown_warning = gr.Markdown("")
            with gr.Row(elem_classes="ct-two-up"):
                token_table = gr.Dataframe(
                    headers=["Position", "Token", "ID"],
                    interactive=False,
                    label="Prompt tokens",
                    elem_classes="ct-table",
                )
                next_token_table = gr.Dataframe(
                    headers=["Rank", "Token", "ID", "Probability"],
                    interactive=False,
                    label="Five most likely next tokens",
                    elem_classes="ct-table",
                )
            gr.Markdown("## Inspect model locations")
            with gr.Row():
                location_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    interactive=True,
                    label="Location",
                )
                token_dropdown = gr.Dropdown(
                    choices=[],
                    value=None,
                    interactive=True,
                    label="Token position",
                )
            inspect_explanation = gr.Markdown(INSPECT_AWAITING)
            inspect_stats = gr.Markdown("")
            heatmap_clip = gr.Checkbox(
                label="Clip heatmap display to the 1st-99th percentile",
                value=False,
            )
            inspect_range = gr.Markdown("")
            with gr.Row(elem_classes="ct-chart-row"):
                magnitude_plot = gr.Plot(
                    label="Token magnitudes",
                    elem_id="ct-magnitude-plot",
                )
                distribution_plot = gr.Plot(label="Selected token distribution")
            heatmap_plot = gr.Plot(label="Token-by-dimension activation heatmap")
            token_click = gr.Textbox(
                value="",
                visible=False,
                elem_id="ct-token-click",
            )
            analyze_button.click(
                fn=lambda prompt: analyze_and_inspect_callback(prompt, manager),
                inputs=prompt_text,
                outputs=[
                    analysis_status,
                    token_count_line,
                    unknown_warning,
                    token_table,
                    next_token_table,
                    location_dropdown,
                    token_dropdown,
                    inspect_explanation,
                    inspect_stats,
                    inspect_range,
                    magnitude_plot,
                    heatmap_plot,
                    distribution_plot,
                    diagram,
                ],
                show_progress="minimal",
            )
            selection_outputs = [
                token_dropdown,
                inspect_explanation,
                inspect_stats,
                inspect_range,
                magnitude_plot,
                heatmap_plot,
                distribution_plot,
                diagram,
            ]
            for dropdown in (location_dropdown, token_dropdown):
                dropdown.change(
                    fn=lambda location, token, clipped: select_location_callback(
                        location, token, clipped, manager
                    ),
                    inputs=[location_dropdown, token_dropdown, heatmap_clip],
                    outputs=selection_outputs,
                    show_progress="minimal",
                )
            heatmap_clip.change(
                fn=lambda location, token, clipped: select_location_callback(
                    location, token, clipped, manager
                ),
                inputs=[location_dropdown, token_dropdown, heatmap_clip],
                outputs=selection_outputs,
                show_progress="minimal",
            )
            token_click.change(
                fn=lambda clicked, location, token, clipped: select_clicked_token_callback(
                    clicked, location, token, clipped, manager
                ),
                inputs=[token_click, location_dropdown, token_dropdown, heatmap_clip],
                outputs=selection_outputs,
                show_progress="minimal",
            )
    return demo


def main() -> None:
    create_app().launch(css=APP_CSS, js=CLICK_BRIDGE_JS, **launch_kwargs())


if __name__ == "__main__":
    main()
