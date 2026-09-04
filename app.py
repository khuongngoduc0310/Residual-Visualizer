import base64
import gc
import json
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
    LOCATIONS,
    InspectionError,
    location_spec,
)
from model import ARCHITECTURE_NAME, ModelConfig


LOGGER = logging.getLogger(__name__)
LOCAL_SERVER_NAME = "127.0.0.1"
SERVER_PORT = 7860
SHARE_PUBLICLY = False
SPA_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"

INSPECT_AWAITING = "Analyze a prompt to capture every internal location."

RESIDUAL_STREAM = "Residual Stream"
ATTENTION = "Attention"
FFN = "FFN"
OUTPUT = "Output"


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


def _model_stages(config: Optional[ModelConfig]) -> list:
    """Ordered model path as plain data for the frontend to render."""
    details = {
        "embedding": (
            None if config is None
            else f"sequence <= {config.max_len}, width {config.embedding_dim}"
        ),
        "attention_update": (
            None if config is None
            else f"{config.num_heads} heads x {config.key_dim}, "
                 f"width {config.embedding_dim}"
        ),
        "attention_residual": "input + update",
        "attention_norm": "post-norm",
        "ffn_hidden": (
            None if config is None else f"hidden width {config.feed_forward_dim}"
        ),
        "ffn_update": (
            None if config is None else f"width {config.embedding_dim}"
        ),
        "ffn_residual": "normalized attention + update",
        "output_norm": "post-norm block output",
        "vocabulary_projection": (
            None if config is None else f"softmax over {config.vocab_size} tokens"
        ),
    }
    names = {
        "embedding": "Token + position embeddings",
        "attention_update": "Causal attention update",
        "attention_residual": "Attention residual addition",
        "attention_norm": "First layer normalization",
        "ffn_hidden": "FFN hidden activation",
        "ffn_update": "FFN update",
        "ffn_residual": "FFN residual addition",
        "output_norm": "Final layer normalization",
        "vocabulary_projection": "Vocabulary projection + softmax",
    }
    categories = {
        "embedding": RESIDUAL_STREAM,
        "attention_update": ATTENTION,
        "attention_residual": RESIDUAL_STREAM,
        "attention_norm": RESIDUAL_STREAM,
        "ffn_hidden": FFN,
        "ffn_update": FFN,
        "ffn_residual": RESIDUAL_STREAM,
        "output_norm": RESIDUAL_STREAM,
        "vocabulary_projection": OUTPUT,
    }
    return [
        {
            "key": key,
            "name": names[key],
            "category": categories[key],
            "detail": details[key],
        }
        for key in names
    ]


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
            "stages": _model_stages(None),
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
        "stages": _model_stages(config),
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


def inspect_payload(
    manager: ModelManager,
    location_key: Optional[str] = None,
    token_position: Optional[int] = None,
    clipped: bool = False,
) -> dict:
    """Re-render inspection views from stored data; never reruns the model."""
    awaiting = {
        "ok": True,
        "state": "awaiting",
        "message": INSPECT_AWAITING,
        "location": None,
        "selected_position": None,
        "token_choices": [],
        "shape": None,
        "capture": None,
        "selected_stats": None,
        "heatmap": None,
        "magnitude": None,
        "distribution": None,
    }
    session = manager.inspection_session
    if session is None:
        return awaiting

    key = location_key or DEFAULT_LOCATION_KEY
    try:
        spec = location_spec(key)
    except InspectionError as error:
        return {**awaiting, "state": "error", "message": str(error)}

    analysis = session.analysis
    token_count = analysis.token_count
    position = (
        token_count - 1
        if token_position is None
        else _clamp_position(token_position, token_count)
    )
    values = analysis.capture.locations[spec.key]
    seq_len, width = values.shape
    labels = [f"{token.position}: {token.text}" for token in analysis.tokens]
    stats = tensor_statistics(values[position])
    lower, upper = display_bounds(values, bool(clipped))
    return {
        "ok": True,
        "state": "ready",
        "message": "",
        "location": {
            "key": spec.key,
            "label": spec.label,
            "category": spec.category,
            "explanation": spec.explanation,
            "normalized": spec.normalized,
        },
        "selected_position": position,
        "token_choices": [
            {"position": token.position, "text": token.text}
            for token in analysis.tokens
        ],
        "shape": {"seq_len": int(seq_len), "width": int(width)},
        "capture": {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        },
        "selected_stats": {
            "norm": float(stats.norm),
            "mean": float(stats.mean),
            "standard_deviation": float(stats.standard_deviation),
            "minimum": float(stats.minimum),
            "maximum": float(stats.maximum),
        },
        "heatmap": {
            "lower": float(lower),
            "upper": float(upper),
            "clipped": bool(clipped),
        },
        "magnitude": _figure_payload(
            render_token_magnitudes(values, labels, position)
        ),
        "heatmap_figure": _figure_payload(
            render_activation_heatmap(values, labels, position, bool(clipped))
        ),
        "distribution": _figure_payload(
            render_token_distribution(
                values[position], position, labels[position]
            )
        ),
    }


def options_payload() -> dict:
    """Static inspection options and the unloaded model path."""
    return {
        "locations": [
            {
                "key": spec.key,
                "label": spec.label,
                "category": spec.category,
                "explanation": spec.explanation,
                "normalized": spec.normalized,
            }
            for spec in LOCATIONS
        ],
        "default_location": DEFAULT_LOCATION_KEY,
        "default_stages": _model_stages(None),
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
            location_key: Optional[str] = None,
            token_position: Optional[int] = None,
            clipped: bool = False,
        ) -> dict:
            return inspect_payload(
                manager, location_key, token_position, clipped
            )

        def get_options(_unused: str = "") -> dict:
            return options_payload()

        gr.api(load_endpoint, api_name="load_checkpoint")
        gr.api(analyze_endpoint, api_name="analyze_prompt")
        gr.api(inspect_endpoint, api_name="inspect_location")
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
