"""Project captured residual states through the output matrix."""

from typing import Optional

import numpy as np

from analysis import PromptAnalysis
from charts import render_readout_delta, render_readout_topk
from checkpoint import LoadedCheckpoint
from inspection import InspectionError
from plotly_json import figure_payload
from readout_view import NEXT_TOKEN_TOP_K, readout_compare, readout_rows


def deembed_probabilities(
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
            "The selected residual state does not match the output projection width."
        )
    if node_key != "output_norm":
        final_norm = checkpoint.model.get_layer("final_output_layer_norm")
        vector = final_norm(vector[None, None, :], training=False).numpy()[0, 0]
    logits = np.matmul(vector, kernel) + bias
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def populate_deembed_payload(
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
    baseline_probabilities = deembed_probabilities(
        baseline_values,
        position,
        checkpoint,
        node_key,
    )
    if view == "ablated" and ablated_analysis is not None:
        ablated_values = ablated_analysis.capture.locations[node_key]
        ablated_probabilities = deembed_probabilities(
            ablated_values,
            position,
            checkpoint,
            node_key,
        )
        compare, highlighted_id = readout_compare(
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
            np.any(np.abs(ablated_values[position] - baseline_values[position]) > 1e-12)
        )
        payload["deembed_figure"] = (
            figure_payload(
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

    rows = readout_rows(
        baseline_probabilities[None, :],
        0,
        checkpoint,
        NEXT_TOKEN_TOP_K,
    )
    payload["deembed_top"] = rows
    payload["deembed_figure"] = figure_payload(
        render_readout_topk(rows, f"{node_label} at {token_label}")
    )
    payload["deembed_present"] = True


__all__ = ["deembed_probabilities", "populate_deembed_payload"]
