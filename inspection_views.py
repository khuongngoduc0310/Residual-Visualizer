"""Assemble the inspection endpoint payload from a stored capture."""

from typing import Optional

import numpy as np

from charts import (
    display_bounds,
    grid_shape,
    render_pattern_heatmap,
    render_token_map_row,
)
from contribution_view import (
    direct_logit_contributions,
    populate_vocab_contribution_payload,
)
from deembed_view import populate_deembed_payload
from engine import ModelManager, ablation_info
from graph_view import node_info_payload, options_payload
from inspection import (
    DEEMBEDDABLE_NODES,
    DEFAULT_NODE_KEY,
    VOCAB_CONTRIBUTABLE_NODES,
    InspectionError,
    node_spec,
)
from inspection_payload import empty_payload
from plotly_json import figure_payload
from readout_view import position_effects, readout_payload

INSPECT_AWAITING = "Analyze a prompt to capture every internal location."


def _clamp_position(position, token_count: int) -> int:
    return max(0, min(int(position), token_count - 1))


def _inspect_node_payload(
    state,
    session,
    node_key: Optional[str],
    token_position: Optional[int],
    view: str,
    highlight_token: Optional[str],
    deembed: bool,
    vocab_contributions: bool,
) -> dict:
    if session is None:
        return empty_payload("awaiting", INSPECT_AWAITING)
    if view not in {"baseline", "ablated", "diff"}:
        return empty_payload("error", f"Unknown inspection view: {view}")
    if view != "baseline" and session.ablated is None:
        return empty_payload(
            "error",
            "No ablation is active. Apply an ablation before changing views.",
            view=view,
        )

    key = node_key or DEFAULT_NODE_KEY
    try:
        spec = node_spec(key)
    except InspectionError as error:
        return empty_payload("error", str(error))

    baseline_analysis = session.analysis
    ablated_analysis = session.ablated.analysis if session.ablated else None
    analysis = baseline_analysis if view == "baseline" else ablated_analysis
    if analysis is None:
        return empty_payload(
            "error",
            "No ablated capture is available.",
            view=view,
            ablation=ablation_info(session.ablated),
        )
    token_count = analysis.token_count
    position = (
        token_count - 1
        if token_position is None
        else _clamp_position(token_position, token_count)
    )
    token_labels = [f"{token.position}: {token.text}" for token in analysis.tokens]
    token_choices = [
        {"position": token.position, "text": token.text} for token in analysis.tokens
    ]
    payload = empty_payload(
        "ready",
        "",
        view=view,
        ablation=ablation_info(session.ablated),
    )
    payload["node"] = node_info_payload(key)
    payload["selected_position"] = position
    payload["token_choices"] = token_choices
    if ablated_analysis is not None:
        payload["position_effects"] = position_effects(
            baseline_analysis.capture.probabilities,
            ablated_analysis.capture.probabilities,
            baseline_analysis,
        )

    kind = spec.kind
    if kind == "readout":
        return readout_payload(
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
            payload["pattern_figure"] = figure_payload(
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
        payload["pattern_figure"] = figure_payload(
            render_pattern_heatmap(values, token_labels, position)
        )
        return payload

    if deembed and key in DEEMBEDDABLE_NODES and view != "diff":
        populate_deembed_payload(
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

    if vocab_contributions and key in VOCAB_CONTRIBUTABLE_NODES and view != "diff":
        populate_vocab_contribution_payload(
            payload,
            state.checkpoint,
            spec.label,
            baseline_analysis,
            ablated_analysis,
            position,
            token_labels[position],
            view,
        )

    tile_rows, tile_cols = grid_shape(width)
    payload["tile"] = {"rows": int(tile_rows), "cols": int(tile_cols)}

    if kind == "hidden" and view != "diff":
        upper = float(np.max(values))
        if upper <= 0.0:
            upper = 1.0
        payload["scale"] = {"lower": 0.0, "upper": upper}
        payload["map_figure"] = figure_payload(
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
        payload["map_figure"] = figure_payload(
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


def inspect_node_payload(
    manager: ModelManager,
    node_key: Optional[str] = None,
    token_position: Optional[int] = None,
    view: str = "baseline",
    highlight_token: Optional[str] = None,
    deembed: bool = False,
    vocab_contributions: bool = False,
) -> dict:
    """Render a stored node capture without rerunning the model."""
    with manager.use_inspection_state() as (state, session):
        try:
            return _inspect_node_payload(
                state,
                session,
                node_key,
                token_position,
                view,
                highlight_token,
                deembed,
                vocab_contributions,
            )
        except InspectionError as error:
            return empty_payload(
                "error",
                str(error),
                view=view,
                ablation=(
                    ablation_info(session.ablated) if session is not None else None
                ),
            )


__all__ = [
    "INSPECT_AWAITING",
    "direct_logit_contributions",
    "inspect_node_payload",
    "options_payload",
]
