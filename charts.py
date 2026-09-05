import math
from typing import Sequence, Tuple

import numpy as np
import plotly.graph_objects as go


def _as_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("values must be a two-dimensional array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("values must not be empty")
    return matrix


def display_bounds(values: np.ndarray) -> Tuple[float, float]:
    """Return the symmetric, zero-centered color bounds of one tensor's own
    data, so each node is normalized by itself."""
    matrix = _as_matrix(values)
    bound = float(np.max(np.abs(matrix)))
    if bound == 0.0:
        bound = 1.0
    return -bound, bound


def _labels(token_labels: Sequence[str], count: int) -> list[str]:
    labels = list(token_labels)
    if len(labels) != count:
        raise ValueError("token_labels must have one label per token")
    return labels


def grid_shape(width: int) -> Tuple[int, int]:
    """Return the near-square (rows, cols) factorization of ``width``.

    Because rows * cols == width there are never empty cells, which keeps the
    resulting figure free of NaN and therefore strict-JSON safe. 256 becomes
    (16, 16); other widths use the factor pair closest to a square."""
    w = int(width)
    if w <= 0:
        raise ValueError("width must be a positive integer")

    root = math.isqrt(w)
    divisors = [d for d in range(1, root + 1) if w % d == 0]
    rows = max(divisors, key=lambda d: min(d, w // d))
    return min(rows, w // rows), max(rows, w // rows)


def render_token_map_row(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_position: int,
    bounds: Tuple[float, float],
    colorscale: str = "RdBu",
    title: str = "Token activation maps",
) -> go.Figure:
    """One heatmap row of per-token square maps.

    Every token's width-256 vector is drawn as its own 16x16 tile and the
    tiles are laid out left to right inside a single Heatmap trace, separated
    by one empty (transparent) column so adjacent tokens stay visually apart.
    One axis pair keeps rendering reliable."""
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    token_count, width = matrix.shape
    tile_rows, tile_cols = grid_shape(width)

    gap = 1 if token_count > 1 else 0
    stride = tile_cols + gap
    columns = stride * token_count - (gap if token_count else 0)

    # Gap cells are left as None so the plot background shows through between
    # tokens; None is strict-JSON safe (unlike NaN).
    z = [[None] * columns for _ in range(tile_rows)]
    dims = [[-1] * columns for _ in range(tile_rows)]
    token_ids = [[-1] * columns for _ in range(tile_rows)]
    for token_index in range(token_count):
        start = token_index * stride
        for row in range(tile_rows):
            for col in range(tile_cols):
                local_dim = row * tile_cols + col
                z[row][start + col] = float(matrix[token_index, local_dim])
                dims[row][start + col] = local_dim
                token_ids[row][start + col] = token_index

    customdata = np.stack(
        [np.asarray(token_ids, dtype=int), np.asarray(dims, dtype=int)],
        axis=-1,
    )
    lower, upper = bounds
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=np.arange(columns),
            y=np.arange(tile_rows),
            zmin=lower,
            zmax=upper,
            colorscale=colorscale,
            colorbar=dict(title="value"),
            customdata=customdata,
            hovertemplate=(
                "token %{customdata[0]} \u00b7 dim %{customdata[1]}"
                "<br>value %{z:.4f}<extra></extra>"
            ),
        )
    )

    if 0 <= selected_position < token_count:
        start = selected_position * stride
        figure.add_shape(
            type="rect",
            x0=start - 0.5,
            x1=start + tile_cols - 0.5,
            y0=-0.5,
            y1=tile_rows - 0.5,
            xref="x",
            yref="y",
            line=dict(color="#dc2626", width=2),
            fillcolor="rgba(0,0,0,0)",
        )

    figure.update_layout(
        title=title,
        xaxis=dict(
            tickvals=[
                token_index * stride + (tile_cols - 1) / 2
                for token_index in range(token_count)
            ],
            ticktext=[label for label in labels],
            tickangle=55,
            tickfont=dict(size=10, color="#475569"),
            side="bottom",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickvals=[],
            showgrid=False,
            zeroline=False,
        ),
        template="plotly_white",
        showlegend=False,
        margin=dict(l=40, r=20, t=70, b=90),
    )
    return figure


def render_pattern_heatmap(
    scores: np.ndarray,
    token_labels: Sequence[str],
    selected_query: int,
    bounds: Tuple[float, float] = (0.0, 1.0),
    colorscale: str = "Blues",
    value_label: str = "weight",
    title: str = "Causal attention pattern (mean over heads)",
) -> go.Figure:
    """Attention pattern: rows are query tokens, columns are key tokens."""
    matrix = _as_matrix(scores)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("attention pattern must be square in sequence")
    labels = _labels(token_labels, matrix.shape[0])
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=np.arange(matrix.shape[1]),
            y=np.arange(matrix.shape[0]),
            zmin=bounds[0],
            zmax=bounds[1],
            colorscale=colorscale,
            colorbar=dict(title=value_label),
            customdata=np.broadcast_to(
                np.arange(matrix.shape[0])[:, None], matrix.shape
            ),
            hovertemplate=(
                f"query %{{y}}<br>key %{{x}}<br>{value_label} %{{z:.4f}}"
                "<extra></extra>"
            ),
        )
    )
    if 0 <= selected_query < matrix.shape[0]:
        figure.add_shape(
            type="rect",
            x0=-0.5,
            x1=matrix.shape[1] - 0.5,
            y0=selected_query - 0.5,
            y1=selected_query + 0.5,
            xref="x",
            yref="y",
            line=dict(color="#dc2626", width=2),
            fillcolor="rgba(0,0,0,0)",
        )
    figure.update_layout(
        title=title,
        xaxis_title="Key position",
        yaxis_title="Query position",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=110, r=20, t=75, b=55),
    )
    return figure


def render_readout_topk(rows: Sequence[dict], token_label: str) -> go.Figure:
    """Horizontal top-K next-token bars for one token."""
    if not rows:
        raise ValueError("readout rows must not be empty")
    texts = [row["text"] for row in rows]
    probabilities = [row["probability"] for row in rows]
    figure = go.Figure(
        go.Bar(
            x=probabilities,
            y=texts,
            orientation="h",
            marker_color="#2563eb",
            text=[f"{probability:.4f}" for probability in probabilities],
            textposition="outside",
            hovertemplate="%{y}<br>probability: %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Most likely next tokens for {token_label}",
        xaxis_title="Probability",
        yaxis_title="Token",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=90, r=55, t=75, b=55),
    )
    return figure


def render_entropy_strip(
    token_labels: Sequence[str],
    entropy: np.ndarray,
    selected_position: int,
) -> go.Figure:
    labels = _labels(token_labels, entropy.shape[0])
    colors = ["#0f766e" if index != selected_position else "#dc2626"
              for index in range(entropy.shape[0])]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=entropy,
            marker_color=colors,
            customdata=np.arange(entropy.shape[0]),
            hovertemplate="%{x}<br>entropy: %{y:.4f} nats<extra></extra>",
        )
    )
    figure.update_layout(
        title="Next-token uncertainty (entropy) per position",
        xaxis_title="Prompt token",
        yaxis_title="Entropy (nats)",
        template="plotly_white",
        margin=dict(l=55, r=20, t=55, b=90),
    )
    return figure


def render_readout_delta(
    rows: Sequence[dict],
    token_label: str,
    highlighted_token_id: int | None = None,
) -> go.Figure:
    """Show the largest ablated-minus-baseline probability movements."""
    if not rows:
        raise ValueError("readout delta rows must not be empty")

    texts = [row["text"] for row in rows]
    deltas = [float(row["delta"]) for row in rows]
    colors = []
    for row, delta in zip(rows, deltas):
        if highlighted_token_id is not None and row["token_id"] == highlighted_token_id:
            colors.append("#d97706")
        else:
            colors.append("#dc2626" if delta < 0.0 else "#2563eb")
    customdata = [
        [float(row["baseline_probability"]), float(row["ablated_probability"])]
        for row in rows
    ]
    figure = go.Figure(
        go.Bar(
            x=deltas,
            y=texts,
            orientation="h",
            marker_color=colors,
            customdata=customdata,
            hovertemplate=(
                "%{y}<br>baseline: %{customdata[0]:.6f}"
                "<br>ablated: %{customdata[1]:.6f}"
                "<br>delta: %{x:.6f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"Ablation effect on next-token probabilities for {token_label}",
        xaxis_title="Ablated - baseline probability",
        yaxis_title="Token",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=90, r=35, t=75, b=55),
    )
    return figure
