"""Focused round-trip and contract tests for raw export artifacts.

These tests stay free of TensorFlow and Gradio so they run quickly and never
touch a model. They build plain :class:`export.ExportContext` objects directly
and assert numeric fidelity, deterministic ordering, metadata completeness,
valid signed deltas, padded-cell exclusion, incompatible-delta refusal, and
non-finite JSON validity.
"""

import io
import json
import math
import zipfile

import numpy as np
import pytest

import export as ex


def make_tokens(count, prefix="tok"):
    return tuple(
        {
            "position": position,
            "text": f"{prefix}-{position}",
            "token_id": 10 + position,
        }
        for position in range(count)
    )


def make_context(
    count=3,
    width=5,
    values=None,
    selected=(0, 2),
    pinned=(1, 2),
    location_key="output_norm",
    location_label="Final block output",
    comparison=None,
    config=None,
):
    if values is None:
        values = np.arange(count * width, dtype=float).reshape(count, width) - 3.0
    return ex.ExportContext(
        checkpoint_path=r"C:\checkpoints\demo-v2",
        architecture="one_block_post_norm_causal_lm",
        device="CPU",
        config=config
        or {
            "vocab_size": 8,
            "max_len": 6,
            "embedding_dim": width,
            "num_heads": 2,
            "key_dim": 4,
            "feed_forward_dim": 12,
        },
        processed_prompt=" ".join(token["text"] for token in make_tokens(count)),
        tokens=make_tokens(count),
        location_key=location_key,
        location_label=location_label,
        values=values,
        selected_positions=selected,
        pinned_measurement=pinned,
        comparison=comparison,
    )


def nextafter_values(count, width):
    values = np.empty((count, width), dtype=float)
    base = np.array(
        [0.1, np.nextafter(1.0, 2.0), -2.5, np.nextafter(0.1, 0.0), 1e-9, 1e9],
        dtype=float,
    )
    for position in range(count):
        for dimension in range(width):
            values[position, dimension] = base[(position + dimension) % base.size] + position
    return values


def test_csv_selection_round_trip_preserves_exact_raw_values():
    values = nextafter_values(3, 5)
    context = make_context(values=values)
    csv_bytes = ex.build_csv_bytes(context, ex.SCOPE_SELECTION)
    metadata_text, table = ex.parse_csv_content(csv_bytes)
    header, rows = table[0], table[1:]

    assert header == list(ex.CSV_COLUMNS_SINGLE)
    assert len(rows) == 2 * 5
    by_cell = {}
    for row in rows:
        position, text, token_id, location, dimension, raw = row
        by_cell[(int(position), int(dimension))] = float(raw)
        assert text == f"tok-{position}"
        assert location == "output_norm"
    for position in (0, 2):
        for dimension in range(5):
            expected = values[position, dimension]
            actual = by_cell[(position, dimension)]
            assert actual == expected, (position, dimension, actual, expected)
    assert json.loads(metadata_text)["scope"] == ex.SCOPE_SELECTION


def test_csv_rows_are_deterministic_and_sorted_by_token_then_dimension():
    values = nextafter_values(3, 5)
    context = make_context(values=values)
    first = ex.build_csv_bytes(context, ex.SCOPE_LOCATION)
    second = ex.build_csv_bytes(context, ex.SCOPE_LOCATION)
    assert first == second

    _, table = ex.parse_csv_content(first)
    header, rows = table[0], table[1:]
    assert len(rows) == 3 * 5
    keys = [(int(row[0]), int(row[4])) for row in rows]
    assert keys == sorted(keys)
    assert [int(row[0]) for row in rows] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]


def test_npz_preserves_matrices_token_positions_and_embedded_metadata():
    values = nextafter_values(3, 5)
    context = make_context(values=values)
    npz = ex.build_npz_bytes(context, ex.SCOPE_SELECTION)
    arrays, metadata_text = ex.load_npz_bytes(npz)

    np.testing.assert_array_equal(arrays["values"], values[[0, 2]])
    np.testing.assert_array_equal(arrays["token_positions"], np.array([0, 2]))
    metadata = json.loads(metadata_text)
    assert metadata["format"] == ex.FORMAT_NPZ
    assert metadata["scope"] == ex.SCOPE_SELECTION
    assert metadata["checkpoint"]["path"] == r"C:\checkpoints\demo-v2"
    assert metadata["checkpoint"]["architecture"] == "one_block_post_norm_causal_lm"
    assert [token["text"] for token in metadata["prompt"]["tokens"]] == [
        "tok-0", "tok-1", "tok-2"
    ]
    assert metadata["location"]["key"] == "output_norm"
    assert metadata["location"]["shape"] == [3, 5]
    assert metadata["source_dimensions"]["indices"] == [0, 1, 2, 3, 4]
    assert metadata["selection"]["positions"] == [0, 2]
    expected_pin = float(values[1, 2])
    assert metadata["pinned_measurement"] == {
        "token_position": 1,
        "dimension": 2,
        "value": expected_pin,
    }


def test_csv_metadata_is_complete_for_single_location_scope():
    context = make_context(width=4, pinned=(0, 1))
    csv_bytes = ex.build_csv_bytes(context, ex.SCOPE_LOCATION)
    metadata_text, _ = ex.parse_csv_content(csv_bytes)
    metadata = json.loads(metadata_text)

    assert metadata["schema"] == ex.EXPORT_METADATA_SCHEMA
    assert metadata["raw"] == {
        "display_clipping": "none",
        "square_padding": "none",
        "description": metadata["raw"]["description"],
    }
    assert metadata["checkpoint"]["config"]["embedding_dim"] == 4
    assert metadata["prompt"]["token_count"] == 3
    assert metadata["prompt"]["processed"] == "tok-0 tok-1 tok-2"
    assert len(metadata["prompt"]["tokens"]) == 3
    assert all(
        set(token) == {"position", "text", "token_id"}
        for token in metadata["prompt"]["tokens"]
    )
    assert metadata["selection"] == {
        "positions": [0, 2],
        "count": 2,
        "scope_note": metadata["selection"]["scope_note"],
    }
    assert "No selection filter" in metadata["selection"]["scope_note"]
    assert metadata["location"]["key"] == "output_norm"
    assert metadata["source_dimensions"] == {
        "indexed_from": 0,
        "count": 4,
        "indices": [0, 1, 2, 3],
    }
    assert metadata["rows"]["count"] == 3 * 4
    assert metadata["arrays"][0]["name"] == "values"
    assert metadata["pinned_measurement"]["token_position"] == 0
    assert metadata["pinned_measurement"]["dimension"] == 1


def test_comparison_export_includes_exact_signed_b_minus_a_delta():
    a_values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    b_values = np.array([[0.5, 2.5], [3.5, -4.0], [5.5, 7.5]])
    context = make_context(
        width=2,
        values=b_values,
        comparison=ex.ExportComparison(
            "attention_update", "Attention update", a_values
        ),
    )

    npz = ex.serialize_raw(context, ex.SCOPE_COMPARISON, ex.FORMAT_NPZ, include_delta=True)
    arrays, metadata_text = ex.load_npz_bytes(npz.content)
    np.testing.assert_array_equal(arrays["values_a"], a_values)
    np.testing.assert_array_equal(arrays["values_b"], b_values)
    np.testing.assert_array_equal(arrays["delta"], b_values - a_values)
    metadata = json.loads(metadata_text)
    assert metadata["delta"]["available"] is True
    assert metadata["delta"]["reason"] == ""
    assert metadata["delta"]["side"] == "B - A"
    assert [loc["side"] for loc in metadata["locations"]] == ["A", "B"]
    assert npz.filename == "circuit-tracer_export_comparison.npz"

    csv = ex.serialize_raw(context, ex.SCOPE_COMPARISON, ex.FORMAT_CSV, include_delta=True)
    metadata_text, table = ex.parse_csv_content(csv.content)
    header, rows = table[0], table[1:]
    assert header == list(ex.CSV_COLUMNS_COMPARISON)
    assert len(rows) == 3 * (2 + 2 + 2)
    sides = {}
    delta_rows = [row for row in rows if row[3] == ex.DELTA_SIDE]
    assert len(delta_rows) == 3 * 2
    for row in delta_rows:
        position, dimension = int(row[0]), int(row[5])
        assert float(row[6]) == b_values[position, dimension] - a_values[position, dimension]
    assert json.loads(metadata_text)["delta"]["available"] is True


def test_incompatible_comparison_refuses_delta_but_still_exports_raw_pair():
    a_values = np.ones((3, 12), dtype=float)
    b_values = np.ones((3, 5), dtype=float)
    context = make_context(
        width=5,
        values=b_values,
        comparison=ex.ExportComparison(
            "ffn_hidden", "FFN hidden activation", a_values
        ),
    )

    with pytest.raises(ex.ExportError) as excinfo:
        ex.serialize_raw(context, ex.SCOPE_COMPARISON, ex.FORMAT_NPZ, include_delta=True)
    message = str(excinfo.value)
    assert "refused" in message
    assert "FFN hidden activation" in message
    assert "3 x 12" in message
    assert "3 x 5" in message
    assert "no broadcasting, truncation, projection, or padding" in message

    npz = ex.serialize_raw(context, ex.SCOPE_COMPARISON, ex.FORMAT_NPZ, include_delta=False)
    arrays, metadata_text = ex.load_npz_bytes(npz.content)
    assert "delta" not in arrays
    assert "values_a" in arrays and "values_b" in arrays
    metadata = json.loads(metadata_text)
    assert metadata["delta"]["available"] is False
    assert "requires exactly equal shapes" in metadata["delta"]["reason"]

    csv = ex.serialize_raw(context, ex.SCOPE_COMPARISON, ex.FORMAT_CSV, include_delta=False)
    _, table = ex.parse_csv_content(csv.content)
    rows = table[1:]
    assert {row[3] for row in rows} == {"A", "B"}
    assert all(row[3] != ex.DELTA_SIDE for row in rows)
    assert len(rows) == 3 * (12 + 5)


def test_square_grid_padding_cells_are_never_exported():
    width = 7  # would pad to a 3x3 square grid in the display only
    values = np.arange(3 * width, dtype=float).reshape(3, width)
    context = make_context(width=width, values=values, selected=(1,))
    metadata_text, table = ex.parse_csv_content(
        ex.build_csv_bytes(context, ex.SCOPE_SELECTION)
    )
    rows = table[1:]
    assert len(rows) == width  # one selected token x source dims only
    assert max(int(row[4]) for row in rows) == width - 1
    assert json.loads(metadata_text)["rows"]["count"] == width
    assert json.loads(metadata_text)["arrays"][0]["shape"] == [1, width]

    npz = ex.build_npz_bytes(context, ex.SCOPE_SELECTION)
    arrays, _ = ex.load_npz_bytes(npz)
    assert arrays["values"].shape == (1, width)
    np.testing.assert_array_equal(arrays["values"][0], values[1])
    assert not np.isnan(arrays["values"]).any()


def test_non_finite_values_keep_csv_parseable_and_json_strictly_valid():
    values = np.array([[np.inf, -np.inf], [np.nan, 3.0]], dtype=float)
    context = make_context(count=2, width=2, values=values, selected=())
    for format_ in (ex.FORMAT_CSV, ex.FORMAT_NPZ):
        if format_ == ex.FORMAT_CSV:
            content = ex.serialize_raw(context, ex.SCOPE_LOCATION, format_).content
            metadata_text, table = ex.parse_csv_content(content)
            raw_values = [float(row[5]) for row in table[1:]]
            assert math.isinf(raw_values[0]) and raw_values[0] > 0
            assert math.isinf(raw_values[1]) and raw_values[1] < 0
            assert math.isnan(raw_values[2])
            assert raw_values[3] == 3.0
        else:
            content = ex.serialize_raw(context, ex.SCOPE_LOCATION, format_).content
            _, metadata_text = ex.load_npz_bytes(content)

        assert "Infinity" not in metadata_text
        assert '"NaN"' not in metadata_text
        metadata = json.loads(metadata_text)
        assert metadata["statistics"]["values"]["maximum"] is None
        assert metadata["statistics"]["values"]["minimum"] is None
        assert metadata["statistics"]["values"]["mean"] is None


def test_scope_validation_gates_selection_and_comparison_availability():
    context = make_context()
    available, _ = ex.scope_reason(context, ex.SCOPE_SELECTION)
    assert available is True
    available, reason = ex.scope_reason(make_context(selected=()), ex.SCOPE_SELECTION)
    assert available is False and "Select one or more" in reason
    available, reason = ex.scope_reason(context, ex.SCOPE_COMPARISON)
    assert available is False and "Pin a location" in reason

    with pytest.raises(ex.ExportError, match="Select one or more"):
        ex.build_csv_bytes(make_context(selected=()), ex.SCOPE_SELECTION)


def test_figure_target_availability_follows_rendered_charts():
    no_selection = make_context(selected=())
    assert ex.figure_reason(no_selection, ex.FIGURE_ACTIVATION)[0] is True
    assert ex.figure_reason(no_selection, ex.FIGURE_MAGNITUDES)[0] is False
    assert ex.figure_reason(no_selection, ex.FIGURE_DISTRIBUTION)[0] is False

    with_selection = make_context(selected=(0,))
    for target in ex.FIGURE_TARGETS:
        assert ex.figure_reason(with_selection, target)[0] is True


def test_formats_and_figure_contracts_are_explicit():
    assert ex.RAW_FORMATS == (ex.FORMAT_CSV, ex.FORMAT_NPZ)
    assert ex.FIGURE_FORMATS == (ex.FORMAT_PNG, ex.FORMAT_SVG)
    assert tuple(ex.FORMAT_LABELS) == ex.EXPORT_FORMATS
    assert tuple(ex.SCOPE_LABELS) == ex.EXPORT_SCOPES
    assert tuple(ex.FIGURE_LABELS) == ex.FIGURE_TARGETS
    assert ex.FIGURE_LABELS[ex.FIGURE_ACTIVATION] == "Active activation view"
    assert ex.FIGURE_LABELS[ex.FIGURE_MAGNITUDES] == "Token magnitudes"
    assert ex.FIGURE_LABELS[ex.FIGURE_DISTRIBUTION] == "Selected-token distribution"


def test_artifact_filenames_are_deterministic_and_scoped():
    assert ex.artifact_filename(ex.SCOPE_LOCATION, ex.FORMAT_CSV, "output_norm") == (
        "circuit-tracer_export_location_output_norm.csv"
    )
    assert ex.artifact_filename(ex.SCOPE_SELECTION, ex.FORMAT_NPZ, "ffn_hidden") == (
        "circuit-tracer_export_selection_ffn_hidden.npz"
    )
    assert ex.artifact_filename(ex.SCOPE_COMPARISON, ex.FORMAT_CSV) == (
        "circuit-tracer_export_comparison.csv"
    )
    with pytest.raises(ex.ExportError):
        ex.artifact_filename(ex.SCOPE_LOCATION, ex.FORMAT_PNG)


def test_npz_archive_embeds_exactly_one_valid_metadata_entry():
    context = make_context()
    npz = ex.build_npz_bytes(context, ex.SCOPE_LOCATION)
    with zipfile.ZipFile(io.BytesIO(npz), mode="r") as archive:
        names = archive.namelist()
        json_entries = [name for name in names if name.endswith(".json")]
        assert json_entries == ["metadata.json"]
        assert "values.npy" in names
    arrays, metadata_text = ex.load_npz_bytes(npz)
    assert json.loads(metadata_text)["format"] == ex.FORMAT_NPZ


def test_csv_and_npz_metadata_share_the_same_context_identity():
    context = make_context(width=4)
    csv = ex.serialize_raw(context, ex.SCOPE_LOCATION, ex.FORMAT_CSV)
    npz = ex.serialize_raw(context, ex.SCOPE_LOCATION, ex.FORMAT_NPZ)
    csv_metadata = json.loads(ex.parse_csv_content(csv.content)[0])
    npz_metadata = json.loads(ex.load_npz_bytes(npz.content)[1])

    for metadata in (csv_metadata, npz_metadata):
        assert metadata["schema"] == ex.EXPORT_METADATA_SCHEMA
        assert metadata["checkpoint"]["path"] == r"C:\checkpoints\demo-v2"
        assert metadata["scope"] == ex.SCOPE_LOCATION
        assert metadata["location"]["key"] == "output_norm"
    assert csv_metadata["checkpoint"] == npz_metadata["checkpoint"]
    assert csv_metadata["prompt"] == npz_metadata["prompt"]
