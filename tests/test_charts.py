import numpy as np
import plotly.graph_objects as go

from charts import (
    display_bounds,
    grid_shape,
    render_entropy_strip,
    render_pattern_heatmap,
    render_readout_topk,
    render_token_map_row,
)


VALUES = np.array([[3.0, 4.0], [-2.0, 1.0], [0.5, -0.5]])
LABELS = ["0: repeat", "1: repeat", "2: final"]


def test_grid_shape_is_an_exact_near_square_factor():
    assert grid_shape(256) == (16, 16)
    assert grid_shape(12) == (3, 4)
    assert grid_shape(8) == (2, 4)
    assert grid_shape(100) == (10, 10)
    assert grid_shape(7) == (1, 7)
    assert grid_shape(1) == (1, 1)


def test_display_bounds_is_symmetric_and_clippable():
    values = np.array([[3.0, 4.0], [-2.0, 1.0], [0.5, -0.5]])
    assert display_bounds(values) == (-4.0, 4.0)
    assert display_bounds(values, clipped=True) == (-3.95, 3.95)


def test_token_map_row_is_a_single_trace_of_concatenated_tiles():
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    figure = render_token_map_row(matrix, LABELS, 1, bounds=(-4.0, 4.0))

    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1
    trace = figure.data[0]
    z = np.asarray(trace.z)
    assert z.shape == (2, 6)  # 2x2 tile per token, 3 tokens side by side
    for token_index in range(3):
        start = token_index * 2
        block = z[:, start : start + 2]
        np.testing.assert_allclose(block, matrix[token_index].reshape(2, 2))
    assert trace.zmin == -4.0
    assert trace.zmax == 4.0
    assert trace.colorbar.title.text == "value"
    assert list(figure.layout.xaxis.ticktext) == LABELS
    rects = [shape for shape in figure.layout.shapes if shape.type == "rect"]
    assert len(rects) == 1  # selected-token outline
    assert rects[0].x0 == 1.5  # block 1 spans columns 2..4 in x coords
    assert rects[0].line.color == "#dc2626"


def test_token_map_row_has_no_padding_for_any_width():
    matrix = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]])
    figure = render_token_map_row(matrix, ["0: abc"], 0, bounds=(0.0, 8.0))

    z = np.asarray(figure.data[0].z)
    assert z.shape == (1, 7)
    assert not np.isnan(z).any()
    np.testing.assert_allclose(z[0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_all_chart_renderers_return_plotly_figures():
    assert isinstance(
        render_token_map_row(VALUES, LABELS, 0, display_bounds(VALUES)),
        go.Figure,
    )
    assert isinstance(
        render_pattern_heatmap(
            np.array([[0.5, 0.5], [0.6, 0.4]]), ["0: x", "1: y"], 0
        ),
        go.Figure,
    )
    rows = [
        {"rank": 1, "text": "a", "probability": 0.9},
        {"rank": 2, "text": "b", "probability": 0.1},
    ]
    assert isinstance(render_readout_topk(rows, "0: x"), go.Figure)
    assert isinstance(
        render_entropy_strip(LABELS, np.array([0.2, 0.4, 0.1]), 1),
        go.Figure,
    )
