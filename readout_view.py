"""Next-token readout rows, comparisons, and the readout payload."""

from typing import Optional

import numpy as np

from analysis import PromptAnalysis, display_text
from charts import render_entropy_strip, render_readout_delta, render_readout_topk
from checkpoint import LoadedCheckpoint
from inspection_payload import empty_payload
from plotly_json import figure_payload

NEXT_TOKEN_TOP_K = 15


def entropy(probabilities: np.ndarray) -> np.ndarray:
    positive = probabilities > 0.0
    logs = np.zeros_like(probabilities)
    logs[positive] = np.log(probabilities[positive])
    return -np.sum(probabilities * logs, axis=1)


def readout_rows(
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


def position_effects(
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


def readout_compare(
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
    baseline_top = readout_rows(baseline, position, checkpoint, top_k)
    ablated_top = readout_rows(ablated, position, checkpoint, top_k)
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


def readout_payload(
    payload: dict,
    checkpoint: LoadedCheckpoint,
    baseline_analysis: PromptAnalysis,
    ablated_analysis: Optional[PromptAnalysis],
    analysis: PromptAnalysis,
    position: int,
    token_labels: list[str],
    view: str,
    highlight_token: Optional[str],
) -> dict:
    baseline_probabilities = baseline_analysis.capture.probabilities
    ablated_probabilities = (
        ablated_analysis.capture.probabilities if ablated_analysis is not None else None
    )
    probabilities = (
        baseline_probabilities if view == "baseline" else ablated_probabilities
    )
    if probabilities is None:
        return empty_payload(
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
    payload["readout_rows"] = readout_rows(
        probabilities, position, checkpoint, NEXT_TOKEN_TOP_K
    )
    if view == "diff" and ablated_probabilities is not None:
        compare, highlighted_id = readout_compare(
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
            figure_payload(
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
        payload["readout_figure"] = figure_payload(
            render_readout_topk(payload["readout_rows"], token_labels[position])
        )
    if ablated_probabilities is not None:
        compare, _ = readout_compare(
            baseline_probabilities,
            ablated_probabilities,
            position,
            checkpoint,
            highlight_token,
        )
        payload["readout_compare"] = compare
        payload["readout_compare_figure"] = (
            figure_payload(
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
        payload["position_effects"] = position_effects(
            baseline_probabilities,
            ablated_probabilities,
            baseline_analysis,
        )
    entropy_values = entropy(probabilities)
    payload["capture"] = {
        "min": float(entropy_values.min()),
        "mean": float(entropy_values.mean()),
        "max": float(entropy_values.max()),
    }
    payload["entropy_figure"] = figure_payload(
        render_entropy_strip(token_labels, entropy_values, position)
    )
    return payload


__all__ = [
    "NEXT_TOKEN_TOP_K",
    "entropy",
    "position_effects",
    "readout_compare",
    "readout_payload",
    "readout_rows",
]
