"""Serialize Plotly figures into plain JSON payloads."""

import base64

import numpy as np


def decode_plotly_json(value):
    """Replace Plotly's base64 typed-array leaves with plain JSON lists."""
    if isinstance(value, dict):
        if isinstance(value.get("bdata"), str) and "dtype" in value:
            array = np.frombuffer(
                base64.b64decode(value["bdata"]),
                dtype=np.dtype(value["dtype"]),
            )
            shape = value.get("shape")
            if isinstance(shape, str):
                shape = [int(dim) for dim in shape.split(",") if dim.strip()]
            if isinstance(shape, list) and shape:
                array = array.reshape(shape)
            return array.tolist()
        return {key: decode_plotly_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_plotly_json(item) for item in value]
    return value


def figure_payload(figure) -> dict:
    """Return a figure as JSON-safe data."""
    return decode_plotly_json(figure.to_plotly_json())


__all__ = ["decode_plotly_json", "figure_payload"]
