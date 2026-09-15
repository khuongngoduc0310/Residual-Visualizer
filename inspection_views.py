"""JSON and chart views over a captured model run."""

import base64
from typing import Optional

import numpy as np

from analysis import PromptAnalysis, display_text
from charts import (
    display_bounds,
    grid_shape,
    render_entropy_strip,
    render_pattern_heatmap,
    render_readout_delta,
    render_readout_topk,
    render_token_map_row,
)
from checkpoint import LoadedCheckpoint
from engine import AblatedResult, ModelManager
from inspection import (
    ABLATABLE_NODES,
    BRANCHES,
    DEEMBEDDABLE_NODES,
    DEFAULT_NODE_KEY,
    EMBEDDING_COMPONENTS,
    SPINE_LINKS,
    SPINE_NODES,
    STREAM_NODES,
    TRACE_ORDER,
    InspectionError,
    node_spec,
)


NEXT_TOKEN_TOP_K = 15
INSPECT_AWAITING = "Analyze a prompt to capture every internal location."


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


def _clamp_position(position, token_count: int) -> int:
    return max(0, min(int(position), token_count - 1))


def _graph_payload() -> dict:
    """Return the declarative wiring of the three-block model."""
    nodes = [
        {
            "key": spec.key,
            "label": spec.label,
            "kind": spec.kind,
            "family": spec.family,
            "explanation": spec.explanation,
            "normalized": spec.normalized,
            "feature_axis": spec.feature_axis,
            "deembeddable": spec.key in DEEMBEDDABLE_NODES,
            "block_index": spec.block_index,
            "stage": spec.stage,
            "width_source": spec.width_source,
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
        "spine": list(SPINE_NODES),
        "spine_links": list(SPINE_LINKS),
        "branches": [
            {
                "key": branch.key,
                "label": branch.label,
                "reads": branch.reads,
                "adds_before": branch.adds_before,
                "path": list(branch.path),
                "observables": list(branch.observables),
                "kind": branch.kind,
                "block_index": branch.block_index,
                "side": branch.side,
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
        "deembeddable": key in DEEMBEDDABLE_NODES,
        "block_index": spec.block_index,
        "stage": spec.stage,
        "width_source": spec.width_source,
        "trace_index": index,
        "trace_count": len(trace),
        "prev_key": trace[index - 1] if index > 0 else None,
        "next_key": trace[index + 1] if index < len(trace) - 1 else None,
    }


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


def _ablation_info(result: Optional[AblatedResult]) -> Optional[dict]:
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


def _position_effects(
    baseline: np.ndarray,
    ablated: np.ndarray,
    analysis: PromptAnalysis,
) -> list:
    effects = 0.5 * np.sum(np.abs(ablated - baseline), axis=1)
    return [
        {
            "position": token.position,
            "text": token.text,
            "effect": float(effects[token.position]),
        }
        for token in analysis.tokens
    ]


def _deembed_probabilities(
    values: np.ndarray,
    position: int,
    checkpoint: LoadedCheckpoint,
    node_key: str,
) -> np.ndarray:
    projection = checkpoint.model.get_layer("token_probabilities")
    weights = projection.get_weights()
    kernel = np.asarray(weights[0])
    bias = (
        np.asarray(weights[1])
        if len(weights) > 1
        else np.zeros(kernel.shape[1], dtype=kernel.dtype)
    )
    vector = np.asarray(values[position])
    if vector.shape != (kernel.shape[0],):
        raise InspectionError(
            "The selected residual state does not match the output projection "
            "width."
        )
    if node_key != "output_norm":
        final_norm = checkpoint.model.get_layer("final_output_layer_norm")
        vector = final_norm(vector[None, None, :], training=False).numpy()[0, 0]
    logits = np.matmul(vector, kernel) + bias
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def _populate_deembed_payload(
    payload: dict,
    checkpoint: LoadedCheckpoint,
    node_label: str,
    baseline_analysis: PromptAnalysis,
    ablated_analysis: Optional[PromptAnalysis],
    position: int,
    token_label: str,
    view: str,
    highlight_token: Optional[str],
) -> None:
    node_key = payload["node"]["key"]
    baseline_values = baseline_analysis.capture.locations[node_key]
    baseline_probabilities = _deembed_probabilities(
        baseline_values,
        position,
        checkpoint,
        node_key,
    )
    if view == "ablated" and ablated_analysis is not None:
        ablated_values = ablated_analysis.capture.locations[node_key]
        ablated_probabilities = _deembed_probabilities(
            ablated_values,
            position,
            checkpoint,
            node_key,
        )
        compare, highlighted_id = _readout_compare(
            baseline_probabilities[None, :],
            ablated_probabilities[None, :],
            0,
            checkpoint,
            highlight_token,
        )
        payload["deembed_top"] = compare["ablated_top"]
        payload["deembed_movers"] = compare["movers"]
        payload["deembed_has_effect"] = compare["has_effect"]
        payload["deembed_state_changed"] = bool(
            np.any(
                np.abs(ablated_values[position] - baseline_values[position])
                > 1e-12
            )
        )
        payload["deembed_figure"] = (
            _figure_payload(
                render_readout_delta(
                    compare["movers"],
                    f"{node_label} at {token_label}",
                    highlighted_token_id=highlighted_id,
                )
            )
            if compare["has_effect"]
            else None
        )
        payload["deembed_present"] = True
        return

    rows = _readout_rows(
        baseline_probabilities[None, :],
        0,
        checkpoint,
        NEXT_TOKEN_TOP_K,
    )
    payload["deembed_top"] = rows
    payload["deembed_figure"] = _figure_payload(
        render_readout_topk(rows, f"{node_label} at {token_label}")
    )
    payload["deembed_present"] = True


def _readout_compare(
    baseline: np.ndarray,
    ablated: np.ndarray,
    position: int,
    checkpoint: LoadedCheckpoint,
    highlight_token: Optional[str],
) -> tuple[dict, Optional[int]]:
    baseline_row = baseline[position]
    ablated_row = ablated[position]
    delta = ablated_row - baseline_row
    highlighted_id = None
    if isinstance(highlight_token, str) and highlight_token.strip():
        candidate = highlight_token.strip()
        try:
            highlighted_id = checkpoint.vocabulary.index(candidate)
        except ValueError:
            highlighted_id = None

    top_k = min(NEXT_TOKEN_TOP_K, checkpoint.config.vocab_size)
    baseline_top = _readout_rows(baseline, position, checkpoint, top_k)
    ablated_top = _readout_rows(ablated, position, checkpoint, top_k)
    candidates = set(np.argsort(-np.abs(delta))[: 2 * top_k].tolist())
    candidates.update(np.argsort(-baseline_row)[:top_k].tolist())
    candidates.update(np.argsort(-ablated_row)[:top_k].tolist())
    if highlighted_id is not None:
        candidates.add(highlighted_id)
    ordered = sorted(
        candidates,
        key=lambda token_id: abs(float(delta[token_id])),
        reverse=True,
    )
    movers = [
        {
            "token_id": int(token_id),
            "text": display_text(int(token_id), checkpoint.vocabulary),
            "baseline_probability": float(baseline_row[token_id]),
            "ablated_probability": float(ablated_row[token_id]),
            "delta": float(delta[token_id]),
            "highlighted": token_id == highlighted_id,
        }
        for token_id in ordered[: max(2 * top_k, 1)]
    ]
    if highlighted_id is not None and not any(
        row["token_id"] == highlighted_id for row in movers
    ):
        movers.append(
            {
                "token_id": int(highlighted_id),
                "text": display_text(highlighted_id, checkpoint.vocabulary),
                "baseline_probability": float(baseline_row[highlighted_id]),
                "ablated_probability": float(ablated_row[highlighted_id]),
                "delta": float(delta[highlighted_id]),
                "highlighted": True,
            }
        )
    return {
        "base_top": baseline_top,
        "ablated_top": ablated_top,
        "movers": movers,
        "has_effect": bool(np.any(np.abs(delta) > 1e-12)),
    }, highlighted_id


def _awaiting_payload(
    state: str,
    message: str,
    view: str = "baseline",
    ablation: Optional[dict] = None,
) -> dict:
    return {
        "ok": True,
        "state": state,
        "message": message,
        "view": view,
        "ablation": ablation,
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
        "readout_compare": None,
        "readout_compare_figure": None,
        "position_effects": [],
        "deembed_present": False,
        "deembed_top": [],
        "deembed_movers": [],
        "deembed_figure": None,
        "deembed_has_effect": None,
        "deembed_state_changed": None,
    }


def _inspect_node_payload(
    state,
    session,
    node_key: Optional[str],
    token_position: Optional[int],
    view: str,
    highlight_token: Optional[str],
    deembed: bool,
) -> dict:
    if session is None:
        return _awaiting_payload("awaiting", INSPECT_AWAITING)
    if view not in {"baseline", "ablated", "diff"}:
        return _awaiting_payload("error", f"Unknown inspection view: {view}")
    if view != "baseline" and session.ablated is None:
        return _awaiting_payload(
            "error",
            "No ablation is active. Apply an ablation before changing views.",
            view=view,
        )

    key = node_key or DEFAULT_NODE_KEY
    try:
        spec = node_spec(key)
    except InspectionError as error:
        return _awaiting_payload("error", str(error))

    baseline_analysis = session.analysis
    ablated_analysis = session.ablated.analysis if session.ablated else None
    analysis = (
        baseline_analysis if view == "baseline" else ablated_analysis
    )
    if analysis is None:
        return _awaiting_payload(
            "error",
            "No ablated capture is available.",
            view=view,
            ablation=_ablation_info(session.ablated),
        )
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
    payload = _awaiting_payload(
        "ready",
        "",
        view=view,
        ablation=_ablation_info(session.ablated),
    )
    payload["node"] = _node_info_payload(key)
    payload["selected_position"] = position
    payload["token_choices"] = token_choices
    if ablated_analysis is not None:
        payload["position_effects"] = _position_effects(
            baseline_analysis.capture.probabilities,
            ablated_analysis.capture.probabilities,
            baseline_analysis,
        )

    kind = spec.kind
    if kind == "readout":
        return _readout_payload(
            payload,
            state.checkpoint,
            baseline_analysis,
            ablated_analysis,
            analysis,
            position,
            token_labels,
            view,
            highlight_token,
        )

    if view == "baseline":
        values = baseline_analysis.capture.locations[key]
    elif view == "ablated":
        values = analysis.capture.locations[key]
    else:
        values = (
            ablated_analysis.capture.locations[key]
            - baseline_analysis.capture.locations[key]
        )
    seq_len, width = values.shape
    payload["shape"] = {"seq_len": int(seq_len), "width": int(width)}
    payload["capture"] = {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }
    payload["figure_kind"] = "hidden" if kind == "hidden" else "activation"
    if view == "diff" and kind == "hidden":
        payload["figure_kind"] = "activation"

    if kind == "pattern":
        payload["figure_kind"] = "pattern"
        if view == "diff":
            lower, upper = display_bounds(values)
            payload["scale"] = {"lower": float(lower), "upper": float(upper)}
            payload["pattern_figure"] = _figure_payload(
                render_pattern_heatmap(
                    values,
                    token_labels,
                    position,
                    bounds=(lower, upper),
                    colorscale="RdBu",
                    value_label="delta",
                    title="Attention pattern difference (ablated - baseline)",
                )
            )
            return payload
        payload["pattern_figure"] = _figure_payload(
            render_pattern_heatmap(values, token_labels, position)
        )
        return payload

    if deembed and key in DEEMBEDDABLE_NODES and view != "diff":
        _populate_deembed_payload(
            payload,
            state.checkpoint,
            spec.label,
            baseline_analysis,
            ablated_analysis,
            position,
            token_labels[position],
            view,
            highlight_token,
        )

    tile_rows, tile_cols = grid_shape(width)
    payload["tile"] = {"rows": int(tile_rows), "cols": int(tile_cols)}

    if kind == "hidden" and view != "diff":
        upper = float(np.max(values))
        if upper <= 0.0:
            upper = 1.0
        payload["scale"] = {"lower": 0.0, "upper": upper}
        payload["map_figure"] = _figure_payload(
            render_token_map_row(
                values,
                token_labels,
                position,
                bounds=(0.0, upper),
                colorscale="Viridis",
                title=f"FFN hidden activation (visible range: 0 to {upper:.4f})",
            )
        )
    else:
        lower, upper = display_bounds(values)
        payload["scale"] = {"lower": float(lower), "upper": float(upper)}
        payload["map_figure"] = _figure_payload(
            render_token_map_row(
                values,
                token_labels,
                position,
                bounds=(lower, upper),
                colorscale="RdBu",
                title=(
                    "Activation difference (ablated - baseline)"
                    if view == "diff"
                    else (
                        f"Token activation maps (visible range: "
                        f"{lower:.4f} to {upper:.4f})"
                    )
                ),
            )
        )

    return payload


def _readout_payload(
    payload,
    checkpoint,
    baseline_analysis,
    ablated_analysis,
    analysis,
    position,
    token_labels,
    view,
    highlight_token,
) -> dict:
    baseline_probabilities = baseline_analysis.capture.probabilities
    ablated_probabilities = (
        ablated_analysis.capture.probabilities
        if ablated_analysis is not None
        else None
    )
    probabilities = (
        baseline_probabilities
        if view == "baseline"
        else ablated_probabilities
    )
    if probabilities is None:
        return _awaiting_payload(
            "error",
            "No ablated capture is available.",
            view=view,
            ablation=payload.get("ablation"),
        )
    payload["shape"] = {
        "seq_len": int(probabilities.shape[0]),
        "width": int(probabilities.shape[1]),
    }
    payload["figure_kind"] = "readout_topk"
    payload["readout_rows"] = _readout_rows(
        probabilities, position, checkpoint, NEXT_TOKEN_TOP_K
    )
    if view == "diff" and ablated_probabilities is not None:
        compare, highlighted_id = _readout_compare(
            baseline_probabilities,
            ablated_probabilities,
            position,
            checkpoint,
            highlight_token,
        )
        payload["figure_kind"] = "readout_delta"
        payload["readout_rows"] = [
            {
                "rank": rank,
                "text": row["text"],
                "token_id": row["token_id"],
                "probability": row["delta"],
            }
            for rank, row in enumerate(compare["movers"], start=1)
        ]
        payload["readout_figure"] = (
            _figure_payload(
                render_readout_delta(
                    compare["movers"],
                    token_labels[position],
                    highlighted_token_id=highlighted_id,
                )
            )
            if compare["movers"]
            else None
        )
    else:
        payload["readout_figure"] = _figure_payload(
            render_readout_topk(payload["readout_rows"], token_labels[position])
        )
    if ablated_probabilities is not None:
        compare, _ = _readout_compare(
            baseline_probabilities,
            ablated_probabilities,
            position,
            checkpoint,
            highlight_token,
        )
        payload["readout_compare"] = compare
        payload["readout_compare_figure"] = (
            _figure_payload(
                render_readout_delta(
                    compare["movers"],
                    token_labels[position],
                    highlighted_token_id=next(
                        (
                            row["token_id"]
                            for row in compare["movers"]
                            if row.get("highlighted")
                        ),
                        None,
                    ),
                )
            )
            if compare["movers"]
            else None
        )
        payload["position_effects"] = _position_effects(
            baseline_probabilities,
            ablated_probabilities,
            baseline_analysis,
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


def inspect_node_payload(
    manager: ModelManager,
    node_key: Optional[str] = None,
    token_position: Optional[int] = None,
    view: str = "baseline",
    highlight_token: Optional[str] = None,
    deembed: bool = False,
) -> dict:
    """Render a stored node capture without rerunning the model."""
    with manager.use_inspection_state() as (state, session):
        return _inspect_node_payload(
            state,
            session,
            node_key,
            token_position,
            view,
            highlight_token,
            deembed,
        )


def options_payload() -> dict:
    """Return the static stream graph and available activation nodes."""
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
        "ablation_nodes": [
            {
                "key": spec.key,
                "label": spec.label,
                "kind": spec.kind,
                "family": spec.family,
            }
            for key in ABLATABLE_NODES
            for spec in (node_spec(key),)
        ],
    }


__all__ = [
    "INSPECT_AWAITING",
    "inspect_node_payload",
    "options_payload",
]
