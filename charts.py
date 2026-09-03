from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
import plotly.graph_objects as go


CLIP_LOW_PERCENTILE = 1.0
CLIP_HIGH_PERCENTILE = 99.0


@dataclass(frozen=True)
class TensorStatistics:
    norm: float
    mean: float
    standard_deviation: float
    minimum: float
    maximum: float


def _as_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must not be empty")
    return matrix


def token_magnitudes(values: np.ndarray) -> np.ndarray:
    """Return one L2 magnitude for each token row."""
    return np.linalg.norm(_as_matrix(values), axis=1)


def tensor_statistics(values: np.ndarray) -> TensorStatistics:
    """Return statistics for one selected token's dimension vector."""
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size == 0:
        raise ValueError("values must not be empty")
    return TensorStatistics(
        norm=float(np.linalg.norm(vector)),
        mean=float(np.mean(vector)),
        standard_deviation=float(np.std(vector)),
        minimum=float(np.min(vector)),
        maximum=float(np.max(vector)),
    )


def display_bounds(values: np.ndarray, clipped: bool = False) -> Tuple[float, float]:
    """Return symmetric color bounds without modifying the source values."""
    matrix = _as_matrix(values)
    if clipped:
        low, high = np.percentile(
            matrix,
            [CLIP_LOW_PERCENTILE, CLIP_HIGH_PERCENTILE],
        )
        bound = max(abs(float(low)), abs(float(high)))
    else:
        bound = float(np.max(np.abs(matrix)))
    if bound == 0.0:
        bound = 1.0
    return -bound, bound


def _labels(token_labels: Sequence[str], count: int) -> list[str]:
    labels = list(token_labels)
    if len(labels) != count:
        raise ValueError("token_labels must have one label per token")
    return labels


def render_token_magnitudes(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_position: int,
) -> go.Figure:
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    colors = ["#2563eb" if index != selected_position else "#dc2626"
              for index in range(matrix.shape[0])]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=token_magnitudes(matrix),
            customdata=np.arange(matrix.shape[0]),
            marker_color=colors,
            hovertemplate="%{x}<br>L2 magnitude: %{y:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Token magnitudes",
        xaxis_title="Prompt token",
        yaxis_title="L2 magnitude",
        template="plotly_white",
        margin=dict(l=55, r=20, t=55, b=90),
    )
    return figure


def render_activation_heatmap(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_position: int,
    clipped: bool = False,
) -> go.Figure:
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    lower, upper = display_bounds(matrix, clipped)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=np.arange(matrix.shape[1]),
            y=labels,
            zmin=lower,
            zmax=upper,
            zmid=0,
            colorscale="RdBu",
            colorbar=dict(title="value"),
            customdata=np.broadcast_to(
                np.arange(matrix.shape[0])[:, None], matrix.shape
            ),
            hovertemplate=(
                "token %{y}<br>dimension %{x}<br>value %{z:.4f}"
                "<extra></extra>"
            ),
        )
    )
    if 0 <= selected_position < matrix.shape[0]:
        figure.add_shape(
            type="rect",
            x0=-0.5,
            x1=matrix.shape[1] - 0.5,
            y0=selected_position - 0.5,
            y1=selected_position + 0.5,
            xref="x",
            yref="y",
            line=dict(color="#dc2626", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
    clipping = "; display clipped to percentile range" if clipped else ""
    figure.update_layout(
        title=f"Activation heatmap (visible range: {lower:.4f} to {upper:.4f}{clipping})",
        xaxis_title="Dimension",
        yaxis_title="Token",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=110, r=20, t=75, b=55),
    )
    return figure


def render_token_distribution(
    values: np.ndarray,
    token_position: int,
    token_label: str,
) -> go.Figure:
    vector = np.asarray(values, dtype=float).reshape(-1)
    stats = tensor_statistics(vector)
    figure = go.Figure(
        go.Histogram(
            x=vector,
            nbinsx=min(30, max(5, int(np.sqrt(vector.size)))),
            marker_color="#0f766e",
            hovertemplate="value: %{x:.4f}<br>count: %{y}<extra></extra>",
        )
    )
    figure.add_vline(
        x=stats.mean,
        line_color="#dc2626",
        line_dash="dash",
        annotation_text=f"mean {stats.mean:.4f}",
    )
    figure.update_layout(
        title=f"Dimension distribution for token {token_position}: {token_label}",
        xaxis_title="Activation value",
        yaxis_title="Dimension count",
        template="plotly_white",
        margin=dict(l=55, r=20, t=75, b=55),
    )
    return figure
