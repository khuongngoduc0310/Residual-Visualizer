from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


CLIP_LOW_PERCENTILE = 1.0
CLIP_HIGH_PERCENTILE = 99.0
HEATMAP_COLUMNS = 4
ACTIVATION_COLORSCALE = [
    [0.0, "#5b21b6"],
    [0.5, "#ffffff"],
    [1.0, "#f97316"],
]
CLIPPING_SUBTITLE = (
    "Display clipped to the 1st-99th percentile; hover values remain raw."
)
SQUARE_DETAIL_SUBTITLE = (
    "Dimensions remain in source order; two-dimensional adjacency is artificial "
    "and gray cells are non-data padding."
)
ACTIVATION_CUSTOMDATA_SCHEMA = "circuit-tracer.activation.v1"
CUSTOMDATA_SCHEMA = 0
CUSTOMDATA_VIEW = 1
CUSTOMDATA_TOKEN_POSITION = 2
CUSTOMDATA_TOKEN_TEXT = 3
CUSTOMDATA_DIMENSION = 4
CUSTOMDATA_RAW_VALUE = 5
OVERVIEW_MIN_HEIGHT = 430
OVERVIEW_ROW_HEIGHT = 26
OVERVIEW_VERTICAL_PADDING = 120
Positions = Union[int, Sequence[int]]


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


def shared_display_bounds(
    values: Union[np.ndarray, Iterable[np.ndarray]],
    *more_values: np.ndarray,
    clipped: bool = False,
) -> Tuple[float, float]:
    """Return one zero-centred range for a collection of captures."""
    if isinstance(values, np.ndarray):
        arrays = [_as_matrix(values), *[_as_matrix(value) for value in more_values]]
    else:
        arrays = [_as_matrix(value) for value in values]
        arrays.extend(_as_matrix(value) for value in more_values)
    if not arrays:
        raise ValueError("values must contain at least one matrix")
    return display_bounds(np.concatenate(arrays, axis=None).reshape(1, -1), clipped)


def signed_comparison(values_a: np.ndarray, values_b: np.ndarray) -> np.ndarray:
    """Return a shape-preserving signed comparison, defined as ``B - A``."""
    a = np.asarray(values_a)
    b = np.asarray(values_b)
    if a.shape != b.shape:
        raise ValueError("values_a and values_b must have the same shape")
    return b - a


def _labels(token_labels: Sequence[str], count: int) -> list[str]:
    labels = list(token_labels)
    if len(labels) != count:
        raise ValueError("token_labels must have one label per token")
    return labels


def _token_text(label: str, position: int) -> str:
    prefix = f"{position}: "
    return label[len(prefix):] if label.startswith(prefix) else label


def _raw_value(value: float) -> str:
    return np.format_float_positional(value, unique=True, trim="-")


def _activation_title(title: str, clipped: bool, subtitle: Optional[str] = None):
    result = {"text": title}
    subtitles = [text for text in (subtitle, CLIPPING_SUBTITLE if clipped else None) if text]
    if subtitles:
        result["subtitle"] = {"text": " ".join(subtitles)}
    return result


def _positions(selected_positions: Positions, count: int) -> list[int]:
    if isinstance(selected_positions, (int, np.integer)):
        selected = [int(selected_positions)]
    else:
        selected = [int(position) for position in selected_positions]
    if not selected or any(position < 0 or position >= count for position in selected):
        raise ValueError("selected_positions must contain valid token positions")
    return list(dict.fromkeys(selected))


def render_token_magnitudes(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_position: Positions,
) -> go.Figure:
    selected = _positions(selected_position, _as_matrix(values).shape[0])
    return _render_magnitudes(values, token_labels, set(selected))


def render_token_magnitudes_all(
    values: np.ndarray,
    token_labels: Sequence[str],
    highlighted_positions: Positions = (),
) -> go.Figure:
    """Render every token's magnitude with an optional highlight subset."""
    matrix = _as_matrix(values)
    if isinstance(highlighted_positions, (int, np.integer)):
        highlighted = {int(highlighted_positions)}
    else:
        highlighted = {int(position) for position in highlighted_positions}
    if any(position < 0 or position >= matrix.shape[0] for position in highlighted):
        raise ValueError("highlighted_positions must contain valid token positions")
    return _render_magnitudes(values, token_labels, highlighted)


def _render_magnitudes(
    values: np.ndarray,
    token_labels: Sequence[str],
    highlighted: set[int],
) -> go.Figure:
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    line_widths = [2 if index in highlighted else 0 for index in range(matrix.shape[0])]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=token_magnitudes(matrix),
            customdata=np.arange(matrix.shape[0]),
            marker=dict(
                color="#2563eb",
                line=dict(color="#111827", width=line_widths),
            ),
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


def _pin_positions(pin_positions: Optional[Positions], count: int) -> list[int]:
    if pin_positions is None:
        return []
    return _positions(pin_positions, count)


def _cell_customdata(
    values: np.ndarray,
    labels: Sequence[str],
    token_positions: Sequence[int],
    view: str,
) -> np.ndarray:
    """Return the stable identity and exact display value for activation cells."""
    rows, width = values.shape
    customdata = np.empty((rows, width, 6), dtype=object)
    customdata[:, :, CUSTOMDATA_SCHEMA] = ACTIVATION_CUSTOMDATA_SCHEMA
    customdata[:, :, CUSTOMDATA_VIEW] = view
    customdata[:, :, CUSTOMDATA_TOKEN_POSITION] = np.asarray(token_positions)[:, None]
    customdata[:, :, CUSTOMDATA_TOKEN_TEXT] = np.asarray(
        [_token_text(labels[position], position) for position in token_positions],
        dtype=object,
    )[:, None]
    customdata[:, :, CUSTOMDATA_DIMENSION] = np.arange(width, dtype=int)[None, :]
    customdata[:, :, CUSTOMDATA_RAW_VALUE] = np.vectorize(
        _raw_value, otypes=[object]
    )(values)
    return customdata


def _measurement_pin(
    measurement_pin: Optional[Tuple[int, int]],
    token_count: int,
    dimension_count: int,
) -> Optional[Tuple[int, int]]:
    if measurement_pin is None:
        return None
    token_position, dimension = map(int, measurement_pin)
    if not (0 <= token_position < token_count and 0 <= dimension < dimension_count):
        return None
    return token_position, dimension


def render_activation_overview(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_positions: Optional[Positions] = None,
    clipped: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
    pin_positions: Optional[Positions] = None,
    location_label: Optional[str] = None,
    measurement_pin: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render all tokens against dimensions in a rectangular overview."""
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    selected = set(_positions(selected_positions, matrix.shape[0])) if selected_positions is not None else set()
    pins = _pin_positions(pin_positions, matrix.shape[0])
    lower, upper = display_bounds(matrix, clipped) if bounds is None else bounds
    dimensions = np.arange(matrix.shape[1], dtype=int)
    customdata = _cell_customdata(
        matrix, labels, range(matrix.shape[0]), "overview"
    )
    figure = go.Figure(go.Heatmap(
        z=matrix,
        x=dimensions,
        y=np.arange(matrix.shape[0]),
        zmin=lower,
        zmax=upper,
        zmid=0,
        colorscale=ACTIVATION_COLORSCALE,
        customdata=customdata,
        hovertemplate=(
            "Token position: %{customdata[2]}<br>"
            "Token text: %{customdata[3]}<br>"
            "Dimension: %{customdata[4]}<br>"
            "Raw value: %{customdata[5]}<extra></extra>"
        ),
    ))
    figure.update_yaxes(
        tickmode="array", tickvals=np.arange(matrix.shape[0]), ticktext=labels,
        autorange="reversed",
    )
    for position in selected:
        figure.add_hrect(y0=position - 0.5, y1=position + 0.5,
                         line_width=2, line_color="#111827",
                         fillcolor="rgba(0,0,0,0)")
    if pins:
        figure.add_trace(go.Scatter(
            x=[-0.5] * len(pins), y=pins, mode="markers",
            marker=dict(symbol="triangle-right", size=10, color="#111827"),
            customdata=np.asarray(pins, dtype=int),
            hovertemplate="pinned token %{customdata}<extra></extra>",
            showlegend=False,
        ))
    measurement = _measurement_pin(
        measurement_pin, matrix.shape[0], matrix.shape[1]
    )
    if measurement is not None:
        token_position, dimension = measurement
        figure.add_trace(go.Scatter(
            x=[dimension], y=[token_position], mode="markers",
            marker=dict(symbol="square-open", size=13, color="#111827", line_width=2),
            customdata=customdata[token_position, dimension][None, :],
            hovertemplate=(
                "Pinned token %{customdata[2]}: %{customdata[3]}<br>"
                "Dimension: %{customdata[4]}<br>"
                "Raw value: %{customdata[5]}<extra></extra>"
            ),
            showlegend=False,
        ))
    figure.update_layout(
        title=_activation_title(location_label or "Activation overview", clipped),
        height=max(
            OVERVIEW_MIN_HEIGHT,
            matrix.shape[0] * OVERVIEW_ROW_HEIGHT + OVERVIEW_VERTICAL_PADDING,
        ),
        template="plotly_white", margin=dict(l=95, r=20, t=55, b=55),
    )
    figure.update_xaxes(title_text="Dimension")
    figure.update_yaxes(title_text="Token")
    return figure


def render_activation_detail(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_positions: Positions,
    mode: str = "square",
    clipped: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
    pin_positions: Optional[Positions] = None,
    location_label: Optional[str] = None,
    measurement_pin: Optional[Tuple[int, int]] = None,
) -> go.Figure:
    """Render selected token vectors as either square grids or indexed rows."""
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    selected = _positions(selected_positions, matrix.shape[0])
    if mode not in {"square", "indexed"}:
        raise ValueError("mode must be 'square' or 'indexed'")
    lower, upper = display_bounds(matrix, clipped) if bounds is None else bounds
    selected_matrix = matrix[selected]
    selected_customdata = _cell_customdata(
        selected_matrix, labels, selected, "detail"
    )
    if mode == "indexed":
        figure = go.Figure(go.Heatmap(
            z=selected_matrix, x=np.arange(matrix.shape[1]), y=np.arange(len(selected)),
            zmin=lower, zmax=upper, zmid=0, colorscale=ACTIVATION_COLORSCALE,
            customdata=selected_customdata,
            hovertemplate=(
                "Token position: %{customdata[2]}<br>"
                "Token text: %{customdata[3]}<br>"
                "Dimension: %{customdata[4]}<br>"
                "Raw value: %{customdata[5]}<extra></extra>"
            ),
        ))
        figure.update_yaxes(tickmode="array", tickvals=np.arange(len(selected)),
                            ticktext=[labels[p] for p in selected], autorange="reversed")
        for row in range(len(selected)):
            figure.add_hrect(
                y0=row - 0.5, y1=row + 0.5, line_width=1,
                line_color="#111827", fillcolor="rgba(0,0,0,0)",
            )
    else:
        side = int(np.ceil(np.sqrt(matrix.shape[1])))
        columns = min(HEATMAP_COLUMNS, len(selected))
        rows = int(np.ceil(len(selected) / columns))
        figure = make_subplots(rows=rows, cols=columns,
                               subplot_titles=[labels[p] for p in selected])
        for tile, (token_position, vector) in enumerate(zip(selected, selected_matrix)):
            grid = np.full(side * side, np.nan)
            grid[:vector.size] = vector
            grid_customdata = np.empty((side * side, 6), dtype=object)
            grid_customdata[:] = (
                ACTIVATION_CUSTOMDATA_SCHEMA, "padding", token_position,
                _token_text(labels[token_position], token_position), -1, "non-data",
            )
            grid_customdata[:vector.size] = selected_customdata[tile]
            row, column = tile // columns + 1, tile % columns + 1
            figure.add_trace(go.Heatmap(
                z=grid.reshape(side, side), x=np.arange(side), y=np.arange(side),
                zmin=lower, zmax=upper, zmid=0, colorscale=ACTIVATION_COLORSCALE,
                customdata=grid_customdata.reshape(side, side, 6),
                hovertemplate=(
                    "Token position: %{customdata[2]}<br>"
                    "Token text: %{customdata[3]}<br>"
                    "Dimension: %{customdata[4]}<br>"
                    "Raw value: %{customdata[5]}<extra></extra>"
                ),
                hoverongaps=False,
                showscale=tile == 0,
            ), row=row, col=column)
            figure.update_xaxes(scaleanchor="y" if tile == 0 else f"y{tile + 1}", row=row, col=column)
            figure.update_yaxes(autorange="reversed", row=row, col=column)
            axis_number = tile + 1
            figure.add_shape(
                type="rect", x0=-0.5, x1=side - 0.5, y0=-0.5, y1=side - 0.5,
                xref="x" if axis_number == 1 else f"x{axis_number}",
                yref="y" if axis_number == 1 else f"y{axis_number}",
                line=dict(color="#111827", width=2),
                fillcolor="rgba(0,0,0,0)",
            )
        figure.update_layout(height=max(430, 300 * rows), plot_bgcolor="#e5e7eb")
    pins = _pin_positions(pin_positions, matrix.shape[0])
    if pins and mode == "indexed":
        for pin in pins:
            if pin in selected:
                figure.add_hline(y=selected.index(pin), line_color="#111827", line_width=2)
    measurement = _measurement_pin(
        measurement_pin, matrix.shape[0], matrix.shape[1]
    )
    if measurement is not None and measurement[0] in selected:
        token_position, dimension = measurement
        point_customdata = selected_customdata[
            selected.index(token_position), dimension
        ][None, :]
        if mode == "indexed":
            figure.add_trace(go.Scatter(
                x=[dimension], y=[selected.index(token_position)], mode="markers",
                marker=dict(symbol="square-open", size=13, color="#111827", line_width=2),
                customdata=point_customdata,
                hovertemplate=(
                    "Pinned token %{customdata[2]}: %{customdata[3]}<br>"
                    "Dimension: %{customdata[4]}<br>"
                    "Raw value: %{customdata[5]}<extra></extra>"
                ),
                showlegend=False,
            ))
        else:
            tile = selected.index(token_position)
            row, column = tile // columns + 1, tile % columns + 1
            figure.add_trace(go.Scatter(
                x=[dimension % side], y=[dimension // side], mode="markers",
                marker=dict(symbol="square-open", size=13, color="#111827", line_width=2),
                customdata=point_customdata,
                hovertemplate=(
                    "Pinned token %{customdata[2]}: %{customdata[3]}<br>"
                    "Dimension: %{customdata[4]}<br>"
                    "Raw value: %{customdata[5]}<extra></extra>"
                ),
                showlegend=False,
            ), row=row, col=column)
    figure.update_layout(
        title=_activation_title(
            location_label or "Activation detail",
            clipped,
            SQUARE_DETAIL_SUBTITLE if mode == "square" else None,
        ),
        template="plotly_white",
    )
    return figure


def render_activation_heatmap(
    values: np.ndarray,
    token_labels: Sequence[str],
    selected_position: Positions,
    clipped: bool = False,
    bounds: Optional[Tuple[float, float]] = None,
    location_label: Optional[str] = None,
) -> go.Figure:
    matrix = _as_matrix(values)
    labels = _labels(token_labels, matrix.shape[0])
    selected = _positions(selected_position, matrix.shape[0])
    matrix = matrix[selected]
    lower, upper = display_bounds(matrix, clipped) if bounds is None else bounds
    side = int(np.ceil(np.sqrt(matrix.shape[1])))
    columns = min(HEATMAP_COLUMNS, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=[labels[position] for position in selected],
        horizontal_spacing=0.06,
        vertical_spacing=min(0.12, 0.8 / max(rows - 1, 1)),
    )
    dimensions = np.arange(side * side, dtype=int).reshape(side, side)
    dimensions.flat[matrix.shape[1] :] = -1
    for tile_position, vector in enumerate(matrix):
        token_position = selected[tile_position]
        grid = np.full(side * side, np.nan)
        grid[: vector.size] = vector
        row = tile_position // columns + 1
        column = tile_position % columns + 1
        figure.add_trace(
            go.Heatmap(
                z=grid.reshape(side, side),
                x=np.arange(side),
                y=np.arange(side),
                zmin=lower,
                zmax=upper,
                zmid=0,
                colorscale=ACTIVATION_COLORSCALE,
                showscale=tile_position == 0,
                customdata=dimensions,
                hovertemplate=(
                    f"token {labels[token_position]}<br>dimension %{{customdata}}<br>"
                    "value %{z:.4f}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        axis_number = tile_position + 1
        xref = "x" if axis_number == 1 else f"x{axis_number}"
        yref = "y" if axis_number == 1 else f"y{axis_number}"
        figure.update_xaxes(
            scaleanchor=yref,
            constrain="domain",
            row=row,
            col=column,
        )
        figure.update_yaxes(
            autorange="reversed",
            row=row,
            col=column,
        )
        figure.add_shape(
            type="rect",
            x0=-0.5,
            x1=side - 0.5,
            y0=-0.5,
            y1=side - 0.5,
            xref=xref,
            yref=yref,
            line=dict(color="#dc2626", width=3),
            fillcolor="rgba(0,0,0,0)",
        )
    for tile_position in range(len(selected), rows * columns):
        row = tile_position // columns + 1
        column = tile_position % columns + 1
        figure.update_xaxes(visible=False, row=row, col=column)
        figure.update_yaxes(visible=False, row=row, col=column)
    clipping = "; display clipped to percentile range" if clipped else ""
    figure.update_layout(
        title=(
            f"{location_label or 'Activation grids by token'} (visible range: {lower:.4f} to "
            f"{upper:.4f}{clipping})"
        ),
        height=max(430, 300 * rows),
        template="plotly_white",
        margin=dict(l=35, r=35, t=75, b=35),
    )
    return figure


def render_token_distribution(
    values: np.ndarray,
    token_position: Optional[int] = None,
    token_label: Optional[str] = None,
    title: Optional[str] = None,
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
        title=title or f"Dimension distribution for token {token_position}: {token_label}",
        xaxis_title="Activation value",
        yaxis_title="Dimension count",
        template="plotly_white",
        margin=dict(l=55, r=20, t=75, b=55),
    )
    return figure
