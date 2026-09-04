import numpy as np
import plotly.graph_objects as go
import pytest

from charts import (
    ACTIVATION_CUSTOMDATA_SCHEMA,
    ACTIVATION_COLORSCALE,
    CLIPPING_SUBTITLE,
    OVERVIEW_ROW_HEIGHT,
    OVERVIEW_VERTICAL_PADDING,
    SQUARE_DETAIL_SUBTITLE,
    display_bounds,
    render_activation_detail,
    render_activation_heatmap,
    render_activation_overview,
    render_token_distribution,
    render_token_magnitudes,
    render_token_magnitudes_all,
    tensor_statistics,
    token_magnitudes,
    shared_display_bounds,
    signed_comparison,
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


def test_heatmap_arranges_selected_dimensions_in_a_square_grid():
    values = np.arange(512, dtype=float).reshape(2, 256)
    labels = ["0: first", "1: second"]

    figure = render_activation_heatmap(values, labels, 1)
    trace = figure.data[0]

    assert isinstance(figure, go.Figure)
    assert trace.z.shape == (16, 16)
    np.testing.assert_array_equal(trace.z, values[1].reshape(16, 16))
    np.testing.assert_array_equal(trace.customdata, np.arange(256).reshape(16, 16))
    assert trace.zmin == -511.0
    assert trace.zmax == 511.0
    assert figure.layout.xaxis.scaleanchor == "y"
    assert any(
        annotation.text == "1: second"
        for annotation in figure.layout.annotations
    )


def test_heatmap_pads_non_square_dimension_vectors():
    figure = render_activation_heatmap(VALUES, LABELS, 0)
    trace = figure.data[0]

    assert trace.z.shape == (2, 2)
    np.testing.assert_array_equal(trace.z[0], [3.0, 4.0])
    assert np.isnan(trace.z[1, 0])
    np.testing.assert_array_equal(trace.customdata, [[0, 1], [-1, -1]])


def test_heatmap_uses_full_capture_for_zero_centered_bounds():
    figure = render_activation_heatmap(VALUES, LABELS, 1)
    trace = figure.data[0]

    assert isinstance(figure, go.Figure)
    np.testing.assert_array_equal(trace.z[0], VALUES[1])
    assert trace.zmin == -2.0
    assert trace.zmax == 2.0
    assert trace.zmid == 0


def test_heatmap_clipping_changes_display_bounds_only():
    figure = render_activation_heatmap(VALUES, LABELS, 0, clipped=True)

    np.testing.assert_array_equal(figure.data[0].z[0], VALUES[0])
    assert display_bounds(VALUES, clipped=True) == (-3.95, 3.95)
    assert "display clipped" in figure.layout.title.text


def test_heatmap_uses_only_the_needed_subplot_columns():
    values = np.arange(1024, dtype=float).reshape(4, 256)
    figure = render_activation_heatmap(values, [str(i) for i in range(4)], [0])

    assert len(figure.data) == 1
    assert figure.layout.xaxis.domain[1] > 0.9


def test_heatmap_supports_maximum_length_token_selection():
    values = np.ones((80, 256), dtype=float)
    figure = render_activation_heatmap(values, [str(i) for i in range(80)], list(range(80)))

    assert len(figure.data) == 80
    assert figure.layout.height == 6000


def test_distribution_accepts_an_explicit_aggregate_title():
    figure = render_token_distribution(
        VALUES[:2].reshape(-1),
        title="Dimension distribution for 2 selected tokens",
    )

    assert figure.layout.title.text == "Dimension distribution for 2 selected tokens"


def test_all_chart_renderers_return_plotly_figures():
    assert isinstance(render_token_magnitudes(VALUES, LABELS, 0), go.Figure)
    assert isinstance(
        render_token_distribution(VALUES[0], 0, LABELS[0]), go.Figure
    )


def test_all_token_magnitudes_render_every_row_with_empty_highlight():
    figure = render_token_magnitudes_all(VALUES, LABELS)
    trace = figure.data[0]

    assert list(trace.x) == LABELS
    np.testing.assert_allclose(trace.y, [5.0, np.sqrt(5), np.sqrt(0.5)])
    assert list(trace.marker.line.width) == [0, 0, 0]


def test_all_token_magnitudes_highlights_only_the_requested_rows():
    figure = render_token_magnitudes_all(VALUES, LABELS, highlighted_positions=[0, 2])
    trace = figure.data[0]

    assert list(trace.x) == LABELS
    assert list(trace.marker.line.width) == [2, 0, 2]
    np.testing.assert_array_equal(trace.customdata, [0, 1, 2])


def test_all_token_magnitudes_accepts_a_single_highlight():
    figure = render_token_magnitudes_all(VALUES, LABELS, highlighted_positions=1)
    assert list(figure.data[0].marker.line.width) == [0, 2, 0]


def test_all_token_magnitudes_rejects_out_of_range_highlight():
    with pytest.raises(ValueError, match="valid token positions"):
        render_token_magnitudes_all(VALUES, LABELS, highlighted_positions=[3])


def test_overview_is_rectangular_and_has_stable_token_dimension_customdata():
    figure = render_activation_overview(VALUES, LABELS, pin_positions=1)
    trace = figure.data[0]

    assert trace.z.shape == VALUES.shape
    assert trace.customdata[2, 1].tolist() == [
        ACTIVATION_CUSTOMDATA_SCHEMA, "overview", 2, "final", 1, "-0.5"
    ]
    assert "Token position: %{customdata[2]}" in trace.hovertemplate
    assert "Token text: %{customdata[3]}" in trace.hovertemplate
    assert "Dimension: %{customdata[4]}" in trace.hovertemplate
    assert "Raw value: %{customdata[5]}" in trace.hovertemplate
    assert figure.data[1].customdata.tolist() == [1]


def test_one_token_overview_preserves_unequal_width_and_exact_raw_hover_value():
    exact_value = np.nextafter(0.1, 1.0)
    values = np.array([[exact_value, -2.0, 3.0, 4.0, 5.0]])

    figure = render_activation_overview(values, ["0: only"])
    trace = figure.data[0]

    assert trace.z.shape == (1, 5)
    np.testing.assert_array_equal(trace.x, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(trace.y, [0])
    assert trace.customdata[0, 0].tolist() == [
        ACTIVATION_CUSTOMDATA_SCHEMA,
        "overview",
        0,
        "only",
        0,
        np.format_float_positional(exact_value, unique=True, trim="-"),
    ]


def test_overview_uses_fixed_diverging_colors_and_safe_all_zero_bounds():
    figure = render_activation_overview(np.zeros((2, 3)), ["0: a", "1: b"])
    trace = figure.data[0]

    assert trace.zmin == -1.0
    assert trace.zmid == 0
    assert trace.zmax == 1.0
    assert list(map(list, trace.colorscale)) == ACTIVATION_COLORSCALE


def test_clipped_overview_preserves_raw_values_and_discloses_clipping():
    values = np.array([[-100.0, -1.0, 0.0], [1.0, 2.0, 100.0]])

    figure = render_activation_overview(values, ["0: first", "1: second"], clipped=True)
    trace = figure.data[0]

    np.testing.assert_array_equal(trace.z, values)
    assert (trace.zmin, trace.zmax) == display_bounds(values, clipped=True)
    assert figure.layout.title.subtitle.text == CLIPPING_SUBTITLE
    assert trace.customdata[0, 0, 5] == "-100"


def test_overview_and_detail_accept_identical_shared_bounds():
    bounds = shared_display_bounds(VALUES, VALUES * 3)
    overview = render_activation_overview(VALUES, LABELS, bounds=bounds)
    detail = render_activation_detail(
        VALUES, LABELS, [1], mode="indexed", bounds=bounds
    )

    assert (overview.data[0].zmin, overview.data[0].zmax) == bounds
    assert (detail.data[0].zmin, detail.data[0].zmax) == bounds


def test_maximum_length_overview_keeps_legible_rows_for_vertical_scrolling():
    values = np.ones((80, 7), dtype=float)
    labels = [f"{position}: token" for position in range(80)]

    figure = render_activation_overview(values, labels)

    assert figure.data[0].z.shape == (80, 7)
    assert figure.layout.height == 80 * OVERVIEW_ROW_HEIGHT + OVERVIEW_VERTICAL_PADDING
    assert len(figure.layout.yaxis.ticktext) == 80


def test_detail_indexed_mode_keeps_token_and_dimension_identity():
    figure = render_activation_detail(VALUES, LABELS, [2, 0], mode="indexed")
    trace = figure.data[0]

    assert trace.z.shape == (2, 2)
    np.testing.assert_array_equal(trace.customdata[:, :, 2], [[2, 2], [0, 0]])
    np.testing.assert_array_equal(trace.customdata[:, :, 4], [[0, 1], [0, 1]])


def test_square_detail_preserves_order_marks_padding_and_discloses_adjacency():
    values = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    figure = render_activation_detail(values, ["0: repeat"], [0])
    trace = figure.data[0]

    assert trace.z.shape == (3, 3)
    np.testing.assert_allclose(trace.z.flat[:5], values[0])
    assert np.isnan(trace.z.flat[5:]).all()
    np.testing.assert_array_equal(trace.customdata.reshape(-1, 6)[:5, 4], range(5))
    assert set(trace.customdata.reshape(-1, 6)[5:, 1]) == {"padding"}
    assert set(trace.customdata.reshape(-1, 6)[5:, 5]) == {"non-data"}
    assert trace.hoverongaps is False
    assert SQUARE_DETAIL_SUBTITLE in figure.layout.title.subtitle.text
    assert figure.layout.plot_bgcolor == "#e5e7eb"


def test_256_dimensions_render_as_an_exact_16_by_16_grid():
    values = np.arange(256, dtype=float).reshape(1, 256)

    figure = render_activation_detail(values, ["0: token"], [0])

    assert figure.data[0].z.shape == (16, 16)
    np.testing.assert_array_equal(figure.data[0].z.reshape(-1), values[0])


def test_square_tiles_can_publish_overview_click_identity():
    figure = render_activation_detail(
        VALUES, LABELS, [0, 1, 2], mode="square", cell_view="overview"
    )

    assert len(figure.data) == 3
    assert [trace.customdata[0, 0, 2] for trace in figure.data] == [0, 1, 2]
    assert all(trace.customdata[0, 0, 1] == "overview" for trace in figure.data)


def test_detail_hover_keeps_exact_raw_values_in_both_modes():
    exact = np.nextafter(0.1, 1.0)
    values = np.array([[exact, -2.0, 3.0]])
    expected = np.format_float_positional(exact, unique=True, trim="-")

    for mode in ("square", "indexed"):
        trace = render_activation_detail(values, ["0: repeat"], [0], mode=mode).data[0]
        assert trace.customdata.reshape(-1, 6)[0].tolist() == [
            ACTIVATION_CUSTOMDATA_SCHEMA, "detail", 0, "repeat", 0, expected
        ]
        assert "Raw value: %{customdata[5]}" in trace.hovertemplate


def test_selection_and_measurement_markers_are_black_structural_outlines():
    overview = render_activation_overview(
        VALUES, LABELS, selected_positions=[1], measurement_pin=(1, 0)
    )
    detail = render_activation_detail(
        VALUES, LABELS, [1], mode="indexed", measurement_pin=(1, 0)
    )

    assert overview.layout.shapes[0].line.color == "#111827"
    assert overview.data[-1].marker.color == "#111827"
    assert overview.data[-1].customdata[0, 2] == 1
    assert overview.data[-1].customdata[0, 4] == 0
    assert detail.data[-1].marker.color == "#111827"
    assert detail.data[-1].customdata[0, 2] == 1
    assert detail.data[-1].customdata[0, 4] == 0


def test_signed_comparison_is_b_minus_a_and_requires_matching_shape():
    np.testing.assert_array_equal(signed_comparison(VALUES, VALUES + 2), np.full_like(VALUES, 2))
    np.testing.assert_allclose(shared_display_bounds(VALUES, VALUES + 10), (-14, 14))

    with np.testing.assert_raises(ValueError):
        signed_comparison(VALUES, VALUES[1:])
