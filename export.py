"""Deterministic raw-data exports for research artifacts.

This module is deliberately free of TensorFlow, Gradio, and Plotly imports so
the round-trip and contract tests stay fast and never touch a model. Exporting
is *pull only*: every builder below produces bytes from an explicit request and
never writes to disk or registers automatic persistence.

Design rules honoured here (and documented in README):

- Values are always the raw captured floats, never display-clipped and never
  taken from square-grid padding cells.
- CSV is deterministic long-form rows, sorted by token position then source
  dimension (and, for comparisons, by a fixed A/B/delta side order).
- NPZ keeps the source matrices as 2-D float64 arrays and embeds a
  ``metadata.json`` entry next to them in the archive.
- Structured metadata uses only finite JSON values; NaN/Infinity are replaced
  with ``null`` so every document stays strictly JSON-valid.
- Comparison exports only carry a signed ``B - A`` delta when A and B have
  exactly equal shapes. Asking for a delta on an incompatible pair raises
  :class:`ExportError` with the precise reason.
"""

import csv
import io
import json
import math
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EXPORT_METADATA_SCHEMA = "circuit-tracer.export.v1"
EXPORT_OPTIONS_SCHEMA = "circuit-tracer.export-options.v1"

# Scopes choose which in-memory research dataset is exported.
SCOPE_SELECTION = "selection"
SCOPE_LOCATION = "location"
SCOPE_COMPARISON = "comparison"
EXPORT_SCOPES = (SCOPE_SELECTION, SCOPE_LOCATION, SCOPE_COMPARISON)
SCOPE_LABELS = {
    SCOPE_SELECTION: "Current selection",
    SCOPE_LOCATION: "Current location",
    SCOPE_COMPARISON: "Comparison",
}
DEFAULT_SCOPE = SCOPE_LOCATION

# Formats choose how the export is delivered.
FORMAT_CSV = "csv"
FORMAT_NPZ = "npz"
FORMAT_PNG = "png"
FORMAT_SVG = "svg"
EXPORT_FORMATS = (FORMAT_CSV, FORMAT_NPZ, FORMAT_PNG, FORMAT_SVG)
FORMAT_LABELS = {
    FORMAT_CSV: "CSV",
    FORMAT_NPZ: "NumPy .npz",
    FORMAT_PNG: "PNG",
    FORMAT_SVG: "SVG",
}
RAW_FORMATS = (FORMAT_CSV, FORMAT_NPZ)
FIGURE_FORMATS = (FORMAT_PNG, FORMAT_SVG)
DEFAULT_FORMAT = FORMAT_CSV

# Figure targets identify which already-rendered Plotly figure is downloaded.
# PNG/SVG exports are delivered browser-side from the rendered figure; the
# module only exposes the contract (order, labels, filename rule) so the
# browser code and its tests stay aligned with the dialog.
FIGURE_ACTIVATION = "activation"
FIGURE_MAGNITUDES = "magnitudes"
FIGURE_DISTRIBUTION = "distribution"
FIGURE_TARGETS = (
    FIGURE_ACTIVATION,
    FIGURE_MAGNITUDES,
    FIGURE_DISTRIBUTION,
)
FIGURE_LABELS = {
    FIGURE_ACTIVATION: "Active activation view",
    FIGURE_MAGNITUDES: "Token magnitudes",
    FIGURE_DISTRIBUTION: "Selected-token distribution",
}
DEFAULT_FIGURE = FIGURE_ACTIVATION

CSV_COLUMNS_SINGLE = (
    "token_position",
    "token_text",
    "token_id",
    "location_key",
    "source_dimension_index",
    "raw_value",
)
CSV_COLUMNS_COMPARISON = (
    "token_position",
    "token_text",
    "token_id",
    "side",
    "location_key",
    "source_dimension_index",
    "raw_value",
)
DELTA_SIDE = "delta"
DELTA_DESCRIPTION = "B - A"


class ExportError(ValueError):
    """An export request is invalid or an incompatible delta was requested."""


@dataclass(frozen=True)
class ExportComparison:
    """The pinned A side of a comparison; B is always the current location."""

    a_location_key: str
    a_location_label: str
    a_values: np.ndarray


@dataclass(frozen=True)
class ExportContext:
    """Plain research context describing one in-memory capture.

    ``tokens`` is a tuple of ``{"position", "text", "token_id"}`` dicts in
    token order. ``values`` is the current location's ``[tokens, width]`` raw
    float matrix. Keeping this a plain dataclass (with no model references)
    lets tests build an ``ExportContext`` without loading TensorFlow.
    """

    checkpoint_path: Optional[str]
    architecture: str
    device: Optional[str]
    config: Dict[str, object]
    processed_prompt: str
    tokens: Tuple[Dict[str, object], ...]
    location_key: str
    location_label: str
    values: np.ndarray
    selected_positions: Tuple[int, ...] = ()
    pinned_measurement: Optional[Tuple[int, int]] = None
    comparison: Optional[ExportComparison] = None

    @property
    def token_count(self) -> int:
        return self.values.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))


def token_rows(tokens: Sequence[Dict[str, object]]) -> Dict[int, Tuple[str, int]]:
    """Index ``(text, token_id)`` by token position for fast CSV lookups."""
    rows = {}
    for token in tokens:
        rows[int(token["position"])] = (str(token["text"]), int(token["token_id"]))
    return rows


def signed_delta(
    a_values: np.ndarray,
    b_values: np.ndarray,
    a_label: str,
    b_label: str,
) -> np.ndarray:
    """Return the exact signed ``B - A`` delta or raise a precise refusal."""
    a = np.asarray(a_values, dtype=float)
    b = np.asarray(b_values, dtype=float)
    if a.shape != b.shape:
        raise ExportError(
            "Delta export refused: A {} has shape {} x {}; B {} has shape "
            "{} x {}. Signed subtraction requires exactly equal shapes; no "
            "broadcasting, truncation, projection, or padding is applied. "
            "Raw A and B remain available for export.".format(
                a_label,
                int(a.shape[0]),
                int(a.shape[1]),
                b_label,
                int(b.shape[0]),
                int(b.shape[1]),
            )
        )
    return b - a


def delta_disabled_reason(
    a_values: np.ndarray,
    b_values: np.ndarray,
    a_label: str,
    b_label: str,
) -> str:
    """Human-readable reason used by metadata when no delta is exported."""
    a = np.asarray(a_values)
    b = np.asarray(b_values)
    return (
        "A {} has shape {} x {}; B {} has shape {} x {}. Signed B - A delta "
        "requires exactly equal shapes, so no delta is exported; no "
        "broadcasting, truncation, projection, or padding is applied. Raw A "
        "and B remain available.".format(
            a_label,
            int(a.shape[0]),
            int(a.shape[1]),
            b_label,
            int(b.shape[0]),
            int(b.shape[1]),
        )
    )


def scope_reason(context: ExportContext, scope: str) -> Tuple[bool, str]:
    """Whether a raw scope is currently valid, with a plain-language reason."""
    if context.values.size == 0:
        return False, "Analyze a prompt to capture a location."
    if scope == SCOPE_SELECTION:
        if not context.selected_positions:
            return False, "Select one or more token positions to export."
        return True, ""
    if scope == SCOPE_LOCATION:
        return True, ""
    if scope == SCOPE_COMPARISON:
        if context.comparison is None:
            return False, "Pin a location as A, then select a different location as B."
        if context.location_key == context.comparison.a_location_key:
            return False, "Select a location other than pinned A to compare."
        return True, ""
    raise ExportError(f"Unknown export scope: {scope}")


def figure_reason(context: ExportContext, target: str) -> Tuple[bool, str]:
    """Whether a figure export target is currently rendered, with a reason."""
    if context.values.size == 0:
        return False, "Analyze a prompt to render charts."
    if target == FIGURE_ACTIVATION:
        return True, ""
    if target == FIGURE_MAGNITUDES:
        if not context.selected_positions:
            return (
                False,
                "The token-magnitudes chart is drawn only when tokens are "
                "selected. Select one or more tokens first.",
            )
        return True, ""
    if target == FIGURE_DISTRIBUTION:
        if not context.selected_positions:
            return (
                False,
                "The selected-token distribution chart is drawn only when "
                "tokens are selected. Select one or more tokens first.",
            )
        return True, ""
    raise ExportError(f"Unknown figure target: {target}")


def _float_text(value: float) -> str:
    """Full-precision CSV text; NaN/inf stay parseable by float()."""
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return np.format_float_positional(number, unique=True, trim="-")


def _validate_context(context: ExportContext) -> None:
    if not isinstance(context, ExportContext):
        raise TypeError("context must be an ExportContext")
    values = np.asarray(context.values)
    if values.ndim != 2:
        raise ExportError("Captured values must be a two-dimensional matrix.")
    if values.shape[0] != len(context.tokens):
        raise ExportError(
            "Captured values token count does not match the prompt token rows."
        )
    invalid = [
        position
        for position in context.selected_positions
        if not (0 <= int(position) < values.shape[0])
    ]
    if invalid:
        raise ExportError(
            "Selected token positions are outside the capture: "
            + ", ".join(str(position) for position in sorted(invalid))
        )


def _comparison_parts(context: ExportContext):
    """Resolve the A/B matrices and the delta availability for the context."""
    a = context.comparison
    a_values = np.asarray(a.a_values, dtype=float)
    b_values = np.asarray(context.values, dtype=float)
    equal = a_values.shape == b_values.shape
    reason = "" if equal else delta_disabled_reason(
        a_values, b_values, a.a_location_label, context.location_label
    )
    return a, a_values, b_values, equal, reason


def _matrix_stats(values: np.ndarray) -> Dict[str, object]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if flat.size == 0:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
            "absolute_maximum": None,
        }
    with np.errstate(all="ignore"):
        return {
            "count": int(flat.size),
            "mean": float(np.mean(flat)),
            "standard_deviation": float(np.std(flat)),
            "minimum": float(np.min(flat)),
            "maximum": float(np.max(flat)),
            "absolute_maximum": float(np.max(np.abs(flat))),
        }


def _pinned_measurement_metadata(context: ExportContext) -> Optional[Dict[str, object]]:
    if context.pinned_measurement is None:
        return None
    token_position, dimension = map(int, context.pinned_measurement)
    values = np.asarray(context.values)
    if not (0 <= token_position < values.shape[0] and 0 <= dimension < values.shape[1]):
        return {
            "token_position": token_position,
            "dimension": dimension,
            "value": None,
        }
    return {
        "token_position": token_position,
        "dimension": dimension,
        "value": _matrix_stats(
            np.asarray([[values[token_position, dimension]]])
        )["mean"],
    }


def _single_metadata_arrays(context: ExportContext, matrix, positions) -> Dict[str, object]:
    width = int(matrix.shape[1])
    return {
        "arrays": [
            {
                "name": "values",
                "shape": [int(matrix.shape[0]), width],
                "dtype": str(np.asarray(matrix).dtype),
                "description": (
                    "The exported raw activation matrix at location key "
                    "'{}'; rows are token positions ({}), columns are source "
                    "dimension indices 0..{} in literal order.".format(
                        context.location_key,
                        ", ".join(str(position) for position in positions),
                        width - 1,
                    )
                ),
            }
        ],
        "location": {
            "key": context.location_key,
            "label": context.location_label,
            "shape": [int(context.values.shape[0]), int(context.values.shape[1])],
        },
        "source_dimensions": {
            "indexed_from": 0,
            "count": width,
            "indices": list(range(width)),
        },
        "rows": {
            "count": int(matrix.shape[0]) * width,
            "order": "sorted by token position, then source dimension index",
        },
    }


def _comparison_metadata_arrays(
    context: ExportContext,
    a,
    a_values,
    include_delta: bool,
    equal: bool,
    reason: str,
) -> Dict[str, object]:
    b_values = np.asarray(context.values, dtype=float)
    arrays = [
        {
            "name": "values_a",
            "shape": [int(a_values.shape[0]), int(a_values.shape[1])],
            "dtype": str(a_values.dtype),
            "description": (
                "Pinned reference A at location key '{}' ({}).".format(
                    a.a_location_key, a.a_location_label
                )
            ),
        },
        {
            "name": "values_b",
            "shape": [int(b_values.shape[0]), int(b_values.shape[1])],
            "dtype": str(b_values.dtype),
            "description": (
                "Current location B at '{}' ({}).".format(
                    context.location_key, context.location_label
                )
            ),
        },
    ]
    if equal:
        arrays.append(
            {
                "name": "delta",
                "shape": [int(b_values.shape[0]), int(b_values.shape[1])],
                "dtype": str(b_values.dtype),
                "description": "Exact signed {} delta {} - {}".format(
                    DELTA_DESCRIPTION,
                    context.location_label,
                    a.a_location_label,
                ),
            }
        )
    delta_entry = {
        "requested": bool(include_delta),
        "available": bool(equal),
        "side": DELTA_DESCRIPTION,
        "label": "{} - {}".format(
            context.location_label, a.a_location_label
        ),
        "reason": reason if not equal else "",
    }
    return {
        "arrays": arrays,
        "locations": [
            {
                "side": "A",
                "key": a.a_location_key,
                "label": a.a_location_label,
                "shape": [int(a_values.shape[0]), int(a_values.shape[1])],
            },
            {
                "side": "B",
                "key": context.location_key,
                "label": context.location_label,
                "shape": [int(b_values.shape[0]), int(b_values.shape[1])],
            },
        ],
        "delta": delta_entry,
        "source_dimensions": {
            "indexed_from": 0,
            "a_count": int(a_values.shape[1]),
            "b_count": int(b_values.shape[1]),
            "a_indices": list(range(int(a_values.shape[1]))),
            "b_indices": list(range(int(b_values.shape[1]))),
        },
        "rows": {
            "count": _comparison_row_count(
                context, a_values, include_delta, equal
            ),
            "order": (
                "sorted by token position, then side order A, B, delta, then "
                "source dimension index"
            ),
        },
    }


def _comparison_row_count(context, a_values, include_delta, equal) -> int:
    token_count = int(context.values.shape[0])
    width_a = int(a_values.shape[1])
    width_b = int(context.values.shape[1])
    count = token_count * (width_a + width_b)
    if equal and include_delta:
        count += token_count * width_a
    return count


def export_metadata(
    context: ExportContext,
    scope: str,
    format_: str,
    *,
    include_delta: bool = False,
) -> Dict[str, object]:
    """Build the complete structured metadata for one raw export request."""
    if format_ not in RAW_FORMATS:
        raise ExportError(
            "Raw export metadata requires a raw format, not '{}'.".format(format_)
        )
    _validate_context(context)
    available, unavailable_reason = scope_reason(context, scope)
    if not available:
        raise ExportError(unavailable_reason or f"Scope '{scope}' is not available.")

    base = {
        "schema": EXPORT_METADATA_SCHEMA,
        "scope": scope,
        "format": format_,
        "raw": {
            "display_clipping": "none",
            "square_padding": "none",
            "description": (
                "Exported values are the raw captured floats. Display "
                "clipping and square-grid padding never alter or appear in "
                "these values."
            ),
        },
        "checkpoint": {
            "path": context.checkpoint_path,
            "architecture": context.architecture,
            "device": context.device,
            "config": dict(context.config),
        },
        "prompt": {
            "processed": context.processed_prompt,
            "token_count": context.token_count,
            "tokens": [dict(token) for token in context.tokens],
        },
        "selection": {
            "positions": sorted(
                {int(position) for position in context.selected_positions}
            ),
            "count": len(context.selected_positions),
        },
        "pinned_measurement": _pinned_measurement_metadata(context),
        "statistics": {"values": _matrix_stats(np.asarray(context.values))},
    }

    if scope == SCOPE_COMPARISON:
        if context.comparison is None:
            raise ExportError(
                "Comparison export requires a pinned A and a different "
                "current location B."
            )
        a, a_values, b_values, equal, reason = _comparison_parts(context)
        if include_delta and not equal:
            raise ExportError(
                "Delta export refused: " + reason
            )
        base.update(
            _comparison_metadata_arrays(
                context, a, a_values, include_delta, equal, reason
            )
        )
        base["statistics"] = {
            "values_a": _matrix_stats(a_values),
            "values_b": _matrix_stats(b_values),
        }
        if equal:
            base["statistics"]["delta"] = _matrix_stats(
                b_values - a_values
            )
        base["selection"] = {
            "positions": sorted(
                {int(position) for position in context.selected_positions}
            ),
            "count": len(context.selected_positions),
            "scope_note": (
                "Comparison export always covers every token position of the "
                "shared capture; the active selection is recorded above for "
                "reference only."
            ),
        }
    else:
        matrix, positions = _single_matrix(context, scope)
        base.update(_single_metadata_arrays(context, matrix, positions))
        base["statistics"]["values_exported"] = _matrix_stats(matrix)
        if scope == SCOPE_SELECTION:
            base["selection"]["scope_note"] = (
                "Rows exported are the selected token positions at the "
                "current location."
            )
        else:
            base["selection"]["scope_note"] = (
                "No selection filter; every captured token position at the "
                "current location is exported. The recorded selection above "
                "is the current workbench state for reference only."
            )
    return base


def _single_matrix(context: ExportContext, scope: str):
    values = np.asarray(context.values, dtype=float)
    if scope == SCOPE_SELECTION:
        positions = sorted(
            {int(position) for position in context.selected_positions}
        )
        return values[list(positions)], positions
    return values, list(range(int(values.shape[0])))


def _metadata_json(metadata: Dict[str, object]) -> str:
    """Strictly-valid JSON: non-finite floats become null, keys sorted."""
    return json.dumps(
        _finite_jsonable(metadata),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _finite_jsonable(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _finite_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_jsonable(item) for item in value]
    return value


def _single_csv_rows(context, matrix, positions):
    text_and_id = token_rows(context.tokens)
    rows = []
    for row_index, position in enumerate(positions):
        text, token_id = text_and_id[int(position)]
        for dimension in range(int(matrix.shape[1])):
            rows.append(
                [
                    position,
                    text,
                    token_id,
                    context.location_key,
                    dimension,
                    _float_text(matrix[row_index, dimension]),
                ]
            )
    return rows


def _comparison_csv_rows(context, a_values, include_delta, equal):
    text_and_id = token_rows(context.tokens)
    a = context.comparison
    b_values = np.asarray(context.values, dtype=float)
    rows = []
    delta = None
    if equal and include_delta:
        delta = b_values - a_values
    delta_location = "{} - {}".format(
        context.location_key, a.a_location_key
    )
    for position in range(int(b_values.shape[0])):
        text, token_id = text_and_id[int(position)]
        for dimension in range(int(a_values.shape[1])):
            rows.append(
                [
                    position,
                    text,
                    token_id,
                    "A",
                    a.a_location_key,
                    dimension,
                    _float_text(a_values[position, dimension]),
                ]
            )
        for dimension in range(int(b_values.shape[1])):
            rows.append(
                [
                    position,
                    text,
                    token_id,
                    "B",
                    context.location_key,
                    dimension,
                    _float_text(b_values[position, dimension]),
                ]
            )
        if delta is not None:
            for dimension in range(int(a_values.shape[1])):
                rows.append(
                    [
                        position,
                        text,
                        token_id,
                        DELTA_SIDE,
                        delta_location,
                        dimension,
                        _float_text(delta[position, dimension]),
                    ]
                )
    return rows


def build_csv_bytes(context: ExportContext, scope: str, *, include_delta: bool = False) -> bytes:
    """Deterministic long-form CSV with a JSON metadata comment header."""
    metadata = export_metadata(
        context, scope, FORMAT_CSV, include_delta=include_delta
    )
    available, reason = scope_reason(context, scope)
    if not available:
        raise ExportError(reason or f"Scope '{scope}' is not available.")

    buffer = io.StringIO(newline="")
    buffer.write("# " + _metadata_json(metadata) + "\n")
    if scope == SCOPE_COMPARISON:
        a_values = np.asarray(context.comparison.a_values, dtype=float)
        b_values = np.asarray(context.values, dtype=float)
        equal = a_values.shape == b_values.shape
        rows = _comparison_csv_rows(context, a_values, include_delta, equal)
        columns = CSV_COLUMNS_COMPARISON
    else:
        matrix, positions = _single_matrix(context, scope)
        rows = _single_csv_rows(context, matrix, positions)
        columns = CSV_COLUMNS_SINGLE
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def build_npz_bytes(context: ExportContext, scope: str, *, include_delta: bool = False) -> bytes:
    """NPZ archive with source matrices and an embedded ``metadata.json``."""
    metadata = export_metadata(
        context, scope, FORMAT_NPZ, include_delta=include_delta
    )
    values = np.asarray(context.values, dtype=float)
    if scope == SCOPE_COMPARISON:
        a_values = np.asarray(context.comparison.a_values, dtype=float)
        arrays = {"values_a": a_values, "values_b": values}
        # Delta is only included when the shapes match exactly and the
        # researcher asked for it; otherwise it is omitted and metadata records
        # the refusal reason.
        if a_values.shape == values.shape and include_delta:
            arrays["delta"] = values - a_values
    else:
        matrix, positions = _single_matrix(context, scope)
        arrays = {"values": matrix}
        if scope == SCOPE_SELECTION:
            arrays["token_positions"] = np.asarray(positions, dtype=np.int64)

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    with zipfile.ZipFile(buffer, mode="a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", _metadata_json(metadata))
    return buffer.getvalue()


def artifact_filename(
    scope: str,
    format_: str,
    location_key: Optional[str] = None,
) -> str:
    """Deterministic download name for one raw artifact."""
    if scope not in EXPORT_SCOPES:
        raise ExportError(f"Unknown export scope: {scope}")
    if format_ not in RAW_FORMATS:
        raise ExportError(f"Not a raw export format: {format_}")
    stem = scope
    if scope == SCOPE_COMPARISON:
        stem = "comparison"
    if location_key and scope in (SCOPE_SELECTION, SCOPE_LOCATION):
        stem = "{}_{}".format(stem, location_key)
    return "circuit-tracer_export_{}.{}".format(stem, format_)


@dataclass(frozen=True)
class RawArtifact:
    filename: str
    media_type: str
    content: bytes
    metadata: Dict[str, object]


def serialize_raw(
    context: ExportContext,
    scope: str,
    format_: str,
    *,
    include_delta: bool = False,
) -> RawArtifact:
    """Build one raw CSV or NPZ artifact, refusing invalid requests."""
    if format_ == FORMAT_CSV:
        content = build_csv_bytes(
            context, scope, include_delta=include_delta
        )
        media_type = "text/csv; charset=utf-8"
    elif format_ == FORMAT_NPZ:
        content = build_npz_bytes(
            context, scope, include_delta=include_delta
        )
        media_type = "application/x-numpy-npz"
    else:
        raise ExportError(
            "serialize_raw supports CSV and NPZ only, not '{}'.".format(format_)
        )
    metadata = export_metadata(
        context, scope, format_, include_delta=include_delta
    )
    return RawArtifact(
        filename=artifact_filename(scope, format_, context.location_key),
        media_type=media_type,
        content=content,
        metadata=metadata,
    )


def parse_csv_content(content: bytes) -> Tuple[str, List[List[str]]]:
    """Return the leading JSON metadata comment and the remaining CSV rows.

    The first non-comment line is the header; rows after it are data rows.
    """
    lines = content.decode("utf-8").splitlines()
    metadata = ""
    index = 0
    while index < len(lines) and lines[index].startswith("# "):
        if not metadata:
            metadata = lines[index][2:]
        index += 1
    reader = csv.reader(lines[index:])
    table = [row for row in reader if row]
    return metadata, table


def load_npz_bytes(content: bytes) -> Tuple[Dict[str, np.ndarray], str]:
    """Return (arrays, metadata_json) for one NPZ artifact."""
    buffer = io.BytesIO(content)
    arrays = np.load(buffer)
    with zipfile.ZipFile(io.BytesIO(content), mode="r") as archive:
        metadata = archive.read("metadata.json").decode("utf-8")
    return {name: arrays[name] for name in arrays.files}, metadata
