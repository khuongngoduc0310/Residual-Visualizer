import numpy as np
import plotly.graph_objects as go

from charts import (
    display_bounds,
    render_activation_heatmap,
    render_token_distribution,
    render_token_magnitudes,
    tensor_statistics,
    token_magnitudes,
)


VALUES = np.array([[3.0, 4.0], [-2.0, 1.0], [0.5, -0.5]])
LABELS = ["0: repeat", "1: repeat", "2: final"]


def test_token_magnitudes_are_row_l2_norms():
    np.testing.assert_allclose(
        token_magnitudes(VALUES), [5.0, np.sqrt(5), np.sqrt(0.5)]
    )


def test_tensor_statistics_match_known_vector():
    stats = tensor_statistics(VALUES[0])

    assert stats.norm == 5.0
    assert stats.mean == 3.5
    assert stats.standard_deviation == 0.5
    assert stats.minimum == 3.0
    assert stats.maximum == 4.0


def test_heatmap_keeps_raw_values_and_uses_zero_centered_bounds():
    figure = render_activation_heatmap(VALUES, LABELS, 1)
    trace = figure.data[0]

    assert isinstance(figure, go.Figure)
    np.testing.assert_array_equal(trace.z, VALUES)
    assert trace.zmin == -4.0
    assert trace.zmax == 4.0
    assert trace.zmid == 0
    assert list(trace.y) == LABELS


def test_heatmap_clipping_changes_display_bounds_only():
    figure = render_activation_heatmap(VALUES, LABELS, 0, clipped=True)

    np.testing.assert_array_equal(figure.data[0].z, VALUES)
    assert display_bounds(VALUES, clipped=True) == (-3.95, 3.95)
    assert "display clipped" in figure.layout.title.text


def test_all_chart_renderers_return_plotly_figures():
    assert isinstance(render_token_magnitudes(VALUES, LABELS, 0), go.Figure)
    assert isinstance(
        render_token_distribution(VALUES[0], 0, LABELS[0]), go.Figure
    )
