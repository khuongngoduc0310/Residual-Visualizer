"""Model lifecycle and application operations.

This module owns the in-memory model session.  It deliberately does not know
about Gradio, FastAPI, or Plotly so the model engine can be tested separately
from the HTTP and visualization adapters.
"""

import gc
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import numpy as np
import tensorflow as tf

from analysis import (
    AnalysisError,
    PromptAnalysis,
    ablate_analysis,
    analyze_prompt,
)
from checkpoint import CheckpointError, LoadedCheckpoint, load_checkpoint
from inspection import (
    AblationError,
    AblationSpec,
    ablation_replacement_values,
    node_spec,
)
from model import ARCHITECTURE_NAME, ModelConfig

LOGGER = logging.getLogger(__name__)


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
    ablated: Optional["AblatedResult"] = None


@dataclass(frozen=True)
class AblatedResult:
    spec: AblationSpec
    baseline_values: list[float]
    analysis: PromptAnalysis


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
            "num_blocks": None,
            "feed_forward_activity_l1": None,
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
        "num_blocks": config.num_blocks,
        "feed_forward_activity_l1": config.feed_forward_activity_l1,
    }


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

    @contextmanager
    def use_inspection_state(
        self,
    ) -> Iterator[tuple[Optional[LoadedState], Optional[InspectionSession]]]:
        """Hold the lifecycle lock while rendering a stored capture."""
        with self._lock:
            yield self._state, self._session

    def store_inspection_session(self, session: InspectionSession) -> None:
        with self._lock:
            self._session = session

    def clear_session(self) -> None:
        """Drop the stored capture without unloading the current model."""
        with self._lock:
            self._session = None

    def store_ablation(self, result: AblatedResult) -> None:
        with self._lock:
            if self._session is None:
                raise CheckpointError("Analyze a prompt before ablating a feature")
            self._session = InspectionSession(
                analysis=self._session.analysis,
                config=self._session.config,
                ablated=result,
            )

    def clear_ablation(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session = InspectionSession(
                    analysis=self._session.analysis,
                    config=self._session.config,
                )

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
    """Tokenize, run, and capture the model once, returning a JSON payload."""
    try:
        with manager.use_loaded_state() as state:
            with tf.device(state.device.tf_device):
                analysis = analyze_prompt(prompt or "", state.checkpoint)
            config = state.checkpoint.config
            # Keep the capture paired with the model that produced it.
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
        "status": (f"Analysis complete for {analysis.token_count} processed token(s)."),
        "token_count": analysis.token_count,
        "max_len": analysis.max_len,
        "unknown_count": analysis.unknown_count,
        "tokens": _tokens_payload(analysis),
        "next_tokens": _next_tokens_payload(analysis),
    }


def ablation_info(result: Optional[AblatedResult]) -> Optional[dict]:
    """Serialize the active ablation for API payloads."""
    if result is None:
        return None
    spec = result.spec
    node = node_spec(spec.node_key)
    return {
        "node_key": spec.node_key,
        "node_label": node.label,
        "dims": list(spec.dims),
        "mode": spec.mode,
        "scope": spec.scope,
        "position": spec.position,
        "baseline_values": result.baseline_values,
    }


def _ablation_failure_payload(message: str) -> dict:
    return {
        "ok": False,
        "status": message,
        "ablation": None,
        "strongest_position": None,
    }


def _ablation_status(
    baseline: PromptAnalysis,
    ablated: PromptAnalysis,
    spec: AblationSpec,
) -> tuple[str, Optional[int]]:
    dim_label = ", ".join(str(dim) for dim in spec.dims)
    dimension_word = "dimension" if len(spec.dims) == 1 else "dimensions"
    dimension_verb = "was" if len(spec.dims) == 1 else "were"
    probability_delta = ablated.capture.probabilities - baseline.capture.probabilities
    position_effect = 0.5 * np.sum(np.abs(probability_delta), axis=1)
    strongest_position = int(np.argmax(position_effect))
    if np.any(position_effect > 1e-12):
        return (
            f"Ablated {spec.node_key} {dimension_word} {dim_label}; strongest "
            f"readout effect is at token position {strongest_position}.",
            strongest_position,
        )

    values = baseline.capture.locations[spec.node_key]
    if spec.scope == "all":
        rows = values
    else:
        rows = values[spec.position : spec.position + 1]
    inactive = bool(np.all(np.abs(rows[:, list(spec.dims)]) <= 1e-12))
    if inactive:
        return (
            f"Ablation produced no measurable change: {spec.node_key} "
            f"{dimension_word} {dim_label} {dimension_verb} inactive at the "
            "ablated token(s).",
            strongest_position,
        )
    return (
        "Ablation produced no measurable probability change; inspect the "
        "per-position effects and readout deltas for this feature.",
        strongest_position,
    )


def ablate_feature_payload(
    manager: ModelManager,
    node_key: str,
    dims: list[int],
    mode: str,
    scope: str,
    position: Optional[int] = None,
) -> dict:
    """Ablate one or more dimensions and store a full comparison capture."""
    try:
        with manager.use_loaded_state() as state:
            session = manager.inspection_session
            if session is None:
                raise AnalysisError("Analyze a prompt before ablating a feature")
            spec = AblationSpec(
                node_key=node_key,
                dims=dims,
                mode=mode,
                scope=scope,
                position=position,
            )
            baseline_values = session.analysis.capture.locations[spec.node_key]
            replacement_values = ablation_replacement_values(
                baseline_values,
                spec,
            )
            token_ids = [token.token_id for token in session.analysis.tokens]
            with tf.device(state.device.tf_device):
                ablated = ablate_analysis(token_ids, state.checkpoint, spec)
            result = AblatedResult(
                spec=spec,
                baseline_values=replacement_values,
                analysis=ablated,
            )
            manager.store_ablation(result)
            status, strongest_position = _ablation_status(
                session.analysis,
                ablated,
                spec,
            )
    except (AblationError, AnalysisError, CheckpointError) as error:
        manager.clear_ablation()
        return _ablation_failure_payload(str(error))
    except Exception:
        manager.clear_ablation()
        LOGGER.exception("Feature ablation failed")
        return _ablation_failure_payload(
            "Feature ablation failed because of an unexpected runtime error."
        )

    return {
        "ok": True,
        "status": status,
        "ablation": ablation_info(result),
        "strongest_position": strongest_position,
    }


def clear_ablation_payload(manager: ModelManager) -> dict:
    manager.clear_ablation()
    return {"ok": True, "status": "Ablation cleared."}


__all__ = [
    "AblatedResult",
    "ComputeDevice",
    "InspectionSession",
    "LoadResult",
    "LoadedState",
    "ModelManager",
    "ablate_feature_payload",
    "ablation_info",
    "analyze_prompt_payload",
    "clear_ablation_payload",
    "config_metadata",
    "detect_compute_device",
    "format_model_summary",
    "load_model_payload",
]
