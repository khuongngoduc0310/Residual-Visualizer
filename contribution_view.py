"""Direct logit attribution for attention and FFN updates."""

from typing import Optional

import numpy as np

from analysis import PromptAnalysis, display_text
from charts import render_vocab_contribution_delta, render_vocab_contributions
from checkpoint import LoadedCheckpoint
from inspection import InspectionError, block_node_key
from plotly_json import figure_payload
from readout_view import NEXT_TOKEN_TOP_K

VOCAB_CONTRIBUTION_TOP_K = 8


def direct_logit_contributions(
    update_values: np.ndarray,
    final_residual_values: np.ndarray,
    position: int,
    checkpoint: LoadedCheckpoint,
) -> np.ndarray:
    projection = checkpoint.model.get_layer("token_probabilities")
    kernel = np.asarray(projection.get_weights()[0], dtype=np.float64)
    update = np.asarray(update_values[position], dtype=np.float64)
    final_residual = np.asarray(final_residual_values[position], dtype=np.float64)
    expected_shape = (kernel.shape[0],)
    if update.shape != expected_shape or final_residual.shape != expected_shape:
        raise InspectionError(
            "The selected update and final residual must match the output "
            "projection width."
        )

    final_norm = checkpoint.model.get_layer("final_output_layer_norm")
    gamma = np.asarray(final_norm.get_weights()[0], dtype=np.float64)
    centered_update = update - np.mean(update)
    centered_final = final_residual - np.mean(final_residual)
    denominator = np.sqrt(
        np.mean(np.square(centered_final)) + float(final_norm.epsilon)
    )
    contributions = np.matmul((centered_update / denominator) * gamma, kernel)
    if not np.all(np.isfinite(contributions)):
        raise InspectionError("Vocabulary contributions were not finite.")
    return np.asarray(contributions)


def vocab_contribution_rows(
    contributions: np.ndarray,
    checkpoint: LoadedCheckpoint,
    top_k: int,
) -> tuple[list, list]:
    scores = np.asarray(contributions)
    if scores.shape != (checkpoint.config.vocab_size,):
        raise InspectionError(
            "Vocabulary contributions do not match the checkpoint vocabulary."
        )

    promoted_ids = sorted(
        np.flatnonzero(scores > 0.0).tolist(),
        key=lambda token_id: (-float(scores[token_id]), token_id),
    )[:top_k]
    suppressed_ids = sorted(
        np.flatnonzero(scores < 0.0).tolist(),
        key=lambda token_id: (float(scores[token_id]), token_id),
    )[:top_k]

    def rows(token_ids: list[int]) -> list:
        return [
            {
                "rank": rank,
                "text": display_text(token_id, checkpoint.vocabulary),
                "token_id": token_id,
                "logit_contribution": float(scores[token_id]),
            }
            for rank, token_id in enumerate(token_ids, start=1)
        ]

    return rows(promoted_ids), rows(suppressed_ids)


def vocab_contribution_movers(
    baseline: np.ndarray,
    ablated: np.ndarray,
    checkpoint: LoadedCheckpoint,
) -> list:
    delta = np.asarray(ablated, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    if not np.all(np.isfinite(delta)):
        raise InspectionError("Vocabulary contribution changes were not finite.")
    if not np.any(np.abs(delta) > 1e-12):
        return []
    top_k = min(NEXT_TOKEN_TOP_K, checkpoint.config.vocab_size)
    ordered = sorted(
        range(checkpoint.config.vocab_size),
        key=lambda token_id: (-abs(float(delta[token_id])), token_id),
    )[:top_k]
    return [
        {
            "token_id": token_id,
            "text": display_text(token_id, checkpoint.vocabulary),
            "baseline_contribution": float(baseline[token_id]),
            "ablated_contribution": float(ablated[token_id]),
            "delta": float(delta[token_id]),
        }
        for token_id in ordered
    ]


def populate_vocab_contribution_payload(
    payload: dict,
    checkpoint: LoadedCheckpoint,
    node_label: str,
    baseline_analysis: PromptAnalysis,
    ablated_analysis: Optional[PromptAnalysis],
    position: int,
    token_label: str,
    view: str,
) -> None:
    node_key = payload["node"]["key"]
    final_residual_key = block_node_key(
        checkpoint.config.num_blocks - 1, "ffn_residual"
    )
    baseline_contributions = direct_logit_contributions(
        baseline_analysis.capture.locations[node_key],
        baseline_analysis.capture.locations[final_residual_key],
        position,
        checkpoint,
    )
    contributions = baseline_contributions
    if view == "ablated" and ablated_analysis is not None:
        contributions = direct_logit_contributions(
            ablated_analysis.capture.locations[node_key],
            ablated_analysis.capture.locations[final_residual_key],
            position,
            checkpoint,
        )
        movers = vocab_contribution_movers(
            baseline_contributions, contributions, checkpoint
        )
        payload["vocab_contribution_movers"] = movers
        payload["vocab_contribution_has_effect"] = bool(movers)
        payload["vocab_contribution_figure"] = (
            figure_payload(
                render_vocab_contribution_delta(
                    movers, f"{node_label} at {token_label}"
                )
            )
            if movers
            else None
        )

    promoted, suppressed = vocab_contribution_rows(
        contributions, checkpoint, VOCAB_CONTRIBUTION_TOP_K
    )
    payload["vocab_contribution_promoted"] = promoted
    payload["vocab_contribution_suppressed"] = suppressed
    if view != "ablated":
        rows = [*promoted, *suppressed]
        payload["vocab_contribution_figure"] = (
            figure_payload(
                render_vocab_contributions(
                    promoted, suppressed, f"{node_label} at {token_label}"
                )
            )
            if rows
            else None
        )
    payload["vocab_contribution_present"] = True


__all__ = [
    "VOCAB_CONTRIBUTION_TOP_K",
    "direct_logit_contributions",
    "populate_vocab_contribution_payload",
    "vocab_contribution_movers",
    "vocab_contribution_rows",
]
