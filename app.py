import base64
import gc
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import gradio as gr
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from analysis import AnalysisError, PromptAnalysis, analyze_prompt, display_text
from checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from charts import (
    display_bounds,
    display_bounds_union,
    grid_shape,
    render_entropy_strip,
    render_pattern_heatmap,
    render_readout_topk,
    render_token_map_row,
)
from inspection import (
    BRANCHES,
    DEFAULT_NODE_KEY,
    EMBEDDING_COMPONENTS,
    SPINE_LINKS,
    SPINE_STATES,
    STREAM_NODES,
    TRACE_ORDER,
    InspectionError,
    node_spec,
)
from model import ARCHITECTURE_NAME, ModelConfig

NEXT_TOKEN_TOP_K = 15
SHARED_FAMILIES = {
    "components",
    "stream_raw",
    "updates",
    "stream_norm",
}

LOGGER = logging.getLogger(__name__)
LOCAL_SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860
SHARE_PUBLICLY = False
SPA_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

INSPECT_AWAITING = "Analyze a prompt to capture every internal location."


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


def config_metadata(config: Optional[ModelConfig]) -> dict:
    if config is None:
        return {
            "architecture": ARCHITECTURE_NAME,
            "path": None,
            "vocab_size": None,
            "max_len": None,
            "embedding_dim": None,
            "num_heads": None,
            "key_dim": None,
            "feed_forward_dim": None,
            "dropout_rate": None,
        }
    return {
        "architecture": ARCHITECTURE_NAME,
        "path": None,
        "vocab_size": config.vocab_size,
        "max_len": config.max_len,
        "embedding_dim": config.embedding_dim,
        "num_heads": config.num_heads,
        "key_dim": config.key_dim,
        "feed_forward_dim": config.feed_forward_dim,
        "dropout_rate": config.dropout_rate,
    }


def _decode_plotly_json(value):
    """Replace Plotly's base64 typed-array leaves with plain JSON lists."""
    if isinstance(value, dict):
        if isinstance(value.get("bdata"), str) and "dtype" in value:
            array = np.frombuffer(
                base64.b64decode(value["bdata"]),
                dtype=np.dtype(value["dtype"]),
            )
            shape = value.get("shape")
            if isinstance(shape, str):
                shape = [int(dim) for dim in shape.split(",") if dim.strip()]
            if isinstance(shape, list) and shape:
                array = array.reshape(shape)
            return array.tolist()
        return {key: _decode_plotly_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_plotly_json(item) for item in value]
    return value


def _figure_payload(figure):
    return _decode_plotly_json(figure.to_plotly_json())


def _tokens_payload(analysis: PromptAnalysis) -> list:
    return [
        {
            "position": token.position,
            "text": token.text,
            "token_id": token.token_id,
        }
        for token in analysis.tokens
    ]


def _next_tokens_payload(analysis: PromptAnalysis) -> list:
    return [
        {
            "rank": token.rank,
            "text": token.text,
            "token_id": token.token_id,
            "probability": float(token.probability),
        }
        for token in analysis.next_tokens
    ]


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
                return LoadResult(
                    success=False,
                    status="No model loaded. Enter a checkpoint folder path first.",
                )

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
                return LoadResult(success=True, status="Model loaded successfully.")
            except CheckpointError as error:
                return LoadResult(
                    success=False,
                    status=f"No model loaded. Checkpoint could not be loaded: {error}",
                )
            except (OSError, ValueError, RuntimeError) as error:
                LOGGER.exception("Checkpoint loading failed")
                return LoadResult(
                    success=False,
                    status=(
                        "No model loaded. Checkpoint could not be loaded: "
                        f"{type(error).__name__}. "
                        "Check the folder and runtime configuration."
                    ),
                )
            except Exception:
                LOGGER.exception("Unexpected checkpoint loading failure")
                return LoadResult(
                    success=False,
                    status=(
                        "No model loaded. Checkpoint could not be loaded because "
                        "of an unexpected runtime error."
                    ),
                )


def load_model_payload(manager: ModelManager, directory) -> dict:
    """Load a checkpoint and return a JSON payload describing the model."""
    result = manager.load(directory)
    state = manager.loaded_state
    if not result.success or state is None:
        return {
            "ok": False,
            "status": result.status,
            "loaded": False,
            "meta": config_metadata(None),
            "device_label": None,
            "summary": None,
        }
    config = state.checkpoint.config
    meta = config_metadata(config)
    meta["path"] = str(state.checkpoint_path)
    return {
        "ok": True,
        "status": result.status,
        "loaded": True,
        "meta": meta,
        "device_label": state.device.label,
        "summary": format_model_summary(state.checkpoint.model),
    }


def analyze_prompt_payload(manager: ModelManager, prompt: str) -> dict:
    """Tokenize, run and capture the model once, returning a JSON payload."""
    try:
        with manager.use_loaded_state() as state:
            with tf.device(state.device.tf_device):
                analysis = analyze_prompt(prompt or "", state.checkpoint)
            config = state.checkpoint.config
        manager.store_inspection_session(
            InspectionSession(analysis=analysis, config=config)
        )
    except (AnalysisError, CheckpointError) as error:
        manager.clear_session()
        return _analysis_failure_payload(str(error))
    except Exception:
        manager.clear_session()
        LOGGER.exception("Prompt analysis failed")
        return _analysis_failure_payload(
            "Prompt analysis failed because of an unexpected runtime error."
        )
    return {
        "ok": True,
        "status": (
            f"Analysis complete for {analysis.token_count} processed token(s)."
        ),
        "token_count": analysis.token_count,
        "max_len": analysis.max_len,
        "unknown_count": analysis.unknown_count,
        "tokens": _tokens_payload(analysis),
        "next_tokens": _next_tokens_payload(analysis),
    }


def _analysis_failure_payload(message: str) -> dict:
    return {
        "ok": False,
        "status": message,
        "token_count": None,
        "max_len": None,
        "unknown_count": None,
        "tokens": [],
        "next_tokens": [],
    }


def _clamp_position(position, token_count: int) -> int:
    return max(0, min(int(position), token_count - 1))


def _graph_payload() -> dict:
    """Declarative wiring of the one-block model for the frontend to draw."""
    nodes = [
        {
            "key": spec.key,
            "label": spec.label,
            "kind": spec.kind,
            "family": spec.family,
            "explanation": spec.explanation,
            "normalized": spec.normalized,
            "feature_axis": spec.feature_axis,
        }
        for spec in STREAM_NODES
    ]
    trace = list(TRACE_ORDER)
    by_key = {item["key"]: item for item in nodes}
    for index, key in enumerate(trace):
        by_key[key]["trace_index"] = index
        by_key[key]["trace_count"] = len(trace)
        by_key[key]["prev_key"] = trace[index - 1] if index > 0 else None
        by_key[key]["next_key"] = (
            trace[index + 1] if index < len(trace) - 1 else None
        )
    return {
        "nodes": nodes,
        "spine": list(SPINE_STATES),
        "spine_links": list(SPINE_LINKS),
        "branches": [
            {
                "key": branch["key"],
                "label": branch["label"],
                "reads": branch["reads"],
                "adds_before": branch["adds_before"],
                "nodes": list(branch["nodes"]),
            }
            for branch in BRANCHES
        ],
        "components": list(EMBEDDING_COMPONENTS),
        "trace": trace,
        "default_node": DEFAULT_NODE_KEY,
    }


def _node_info_payload(key: str) -> dict:
    spec = node_spec(key)
    trace = list(TRACE_ORDER)
    index = trace.index(key)
    return {
        "key": spec.key,
        "label": spec.label,
        "kind": spec.kind,
        "family": spec.family,
        "explanation": spec.explanation,
        "normalized": spec.normalized,
        "feature_axis": spec.feature_axis,
        "trace_index": index,
        "trace_count": len(trace),
        "prev_key": trace[index - 1] if index > 0 else None,
        "next_key": trace[index + 1] if index < len(trace) - 1 else None,
    }


def _family_arrays(analysis: PromptAnalysis, family: str):
    from inspection import family_keys

    return [
        analysis.capture.locations[key]
        for key in family_keys(family)
        if key in analysis.capture.locations
    ]


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    positive = probabilities > 0.0
    logs = np.zeros_like(probabilities)
    logs[positive] = np.log(probabilities[positive])
    return -np.sum(probabilities * logs, axis=1)


def _readout_rows(
    probabilities: np.ndarray,
    position: int,
    checkpoint: LoadedCheckpoint,
    top_k: int,
) -> list:
    row = probabilities[position]
    k = min(top_k, checkpoint.config.vocab_size)
    top = np.argsort(-row)[:k]
    return [
        {
            "rank": rank,
            "text": display_text(int(token_id), checkpoint.vocabulary),
            "token_id": int(token_id),
            "probability": float(row[token_id]),
        }
        for rank, token_id in enumerate(top, start=1)
    ]


def _awaiting_payload(state: str, message: str) -> dict:
    return {
        "ok": True,
        "state": state,
        "message": message,
        "node": None,
        "selected_position": None,
        "token_choices": [],
        "shape": None,
        "capture": None,
        "scale": None,
        "tile": None,
        "figure_kind": None,
        "map_figure": None,
        "pattern_figure": None,
        "readout_figure": None,
        "entropy_figure": None,
        "readout_rows": [],
    }


def inspect_node_payload(
    manager: ModelManager,
    node_key: Optional[str] = None,
    token_position: Optional[int] = None,
    clipped: bool = False,
) -> dict:
    """Render the chosen stream node from stored captures; never reruns the
    model."""
    session = manager.inspection_session
    if session is None:
        return _awaiting_payload("awaiting", INSPECT_AWAITING)

    key = node_key or DEFAULT_NODE_KEY
    try:
        spec = node_spec(key)
    except InspectionError as error:
        return _awaiting_payload("error", str(error))

    analysis = session.analysis
    token_count = analysis.token_count
    position = (
        token_count - 1
        if token_position is None
        else _clamp_position(token_position, token_count)
    )
    token_labels = [
        f"{token.position}: {token.text}" for token in analysis.tokens
    ]
    token_choices = [
        {"position": token.position, "text": token.text}
        for token in analysis.tokens
    ]
    payload = _awaiting_payload("ready", "")
    payload["node"] = _node_info_payload(key)
    payload["selected_position"] = position
    payload["token_choices"] = token_choices

    kind = spec.kind
    if kind == "readout":
        return _readout_payload(payload, manager, analysis, position, token_labels)

    values = analysis.capture.locations[key]
    seq_len, width = values.shape
    payload["shape"] = {"seq_len": int(seq_len), "width": int(width)}
    payload["capture"] = {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }
    payload["figure_kind"] = "hidden" if kind == "hidden" else "activation"

    if kind == "pattern":
        payload["figure_kind"] = "pattern"
        payload["pattern_figure"] = _figure_payload(
            render_pattern_heatmap(values, token_labels, position)
        )
        return payload

    tile_rows, tile_cols = grid_shape(width)
    payload["tile"] = {
        "rows": int(tile_rows),
        "cols": int(tile_cols),
    }

    if kind == "hidden":
        flat = values.reshape(-1)
        if bool(clipped):
            upper = float(np.percentile(flat, 99.0))
        else:
            upper = float(np.max(flat))
        if upper <= 0.0:
            upper = 1.0
        payload["scale"] = {
            "lower": 0.0,
            "upper": upper,
            "clipped": bool(clipped),
            "family": spec.family,
            "source": "matrix",
        }
        payload["map_figure"] = _figure_payload(
            render_token_map_row(
                values,
                token_labels,
                position,
                bounds=(0.0, upper),
                colorscale="Viridis",
                title=(
                    f"FFN hidden activation (visible range: 0 to {upper:.4f})"
                ),
            )
        )
    else:
        family = spec.family
        shared = family in SHARED_FAMILIES
        if shared:
            lower, upper = display_bounds_union(
                _family_arrays(analysis, family), bool(clipped)
            )
        else:
            lower, upper = display_bounds(values, bool(clipped))
        payload["scale"] = {
            "lower": float(lower),
            "upper": float(upper),
            "clipped": bool(clipped),
            "family": family,
            "source": "family" if shared else "matrix",
        }
        payload["map_figure"] = _figure_payload(
            render_token_map_row(
                values,
                token_labels,
                position,
                bounds=(lower, upper),
                colorscale="RdBu",
                title=(
                    f"Token activation maps (visible range: "
                    f"{lower:.4f} to {upper:.4f}"
                    + (", percentile clipped)" if bool(clipped) else ")")
                ),
            )
        )

    return payload


def _readout_payload(payload, manager, analysis, position, token_labels) -> dict:
    state = manager.loaded_state
    if state is None:
        return _awaiting_payload("error", "Load a checkpoint before the readout.")
    checkpoint = state.checkpoint
    probabilities = analysis.capture.probabilities
    payload["shape"] = {
        "seq_len": int(probabilities.shape[0]),
        "width": int(probabilities.shape[1]),
    }
    payload["figure_kind"] = "readout_topk"
    payload["readout_rows"] = _readout_rows(
        probabilities, position, checkpoint, NEXT_TOKEN_TOP_K
    )
    payload["readout_figure"] = _figure_payload(
        render_readout_topk(payload["readout_rows"], token_labels[position])
    )
    entropy = _entropy(probabilities)
    payload["capture"] = {
        "min": float(entropy.min()),
        "mean": float(entropy.mean()),
        "max": float(entropy.max()),
    }
    payload["entropy_figure"] = _figure_payload(
        render_entropy_strip(token_labels, entropy, position)
    )
    return payload


def options_payload() -> dict:
    """Static stream graph describing the model wiring."""
    return {
        "graph": _graph_payload(),
        "locations": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "family": spec.family,
            }
            for spec in STREAM_NODES
        ],
    }


def create_app(manager: Optional[ModelManager] = None) -> gr.Blocks:
    """Build the Gradio app exposing JSON endpoints; never launches here."""
    manager = manager or ModelManager()
    with gr.Blocks(title="Circuit Tracer") as demo:
        def load_endpoint(path: str) -> dict:
            return load_model_payload(manager, path)

        def analyze_endpoint(prompt: str) -> dict:
            return analyze_prompt_payload(manager, prompt)

        def inspect_endpoint(
            node_key: Optional[str] = None,
            token_position: Optional[int] = None,
            clipped: bool = False,
        ) -> dict:
            return inspect_node_payload(
                manager, node_key, token_position, clipped
            )

        def get_options(_unused: str = "") -> dict:
            return options_payload()

        gr.api(load_endpoint, api_name="load_checkpoint")
        gr.api(analyze_endpoint, api_name="analyze_prompt")
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


if __name__ == "__main__":
    main()
