"""The JSON shape returned by the inspection endpoint.

This module owns the complete set of payload keys. The frontend contract in
``frontend/src/types.ts`` mirrors it and ``tests/test_inspection_payload.py``
pins it, so a schema change happens in one place and fails loudly if it drifts.
"""

from typing import Any, Optional

# Fields every payload carries once a capture exists; ``empty_payload`` resets
# each of them to its "nothing to show" value.
_DEFAULT_FIELDS: dict[str, Any] = {
    "node": None,
    "selected_position": None,
    "token_choices": [],
    "shape": None,
    "capture": None,
    "scale": None,
    "tile": None,
    "figure_kind": None,
    "map_figure": None,
    "pattern_figure": None,
    "readout_figure": None,
    "entropy_figure": None,
    "readout_rows": [],
    "readout_compare": None,
    "readout_compare_figure": None,
    "position_effects": [],
    "deembed_present": False,
    "deembed_top": [],
    "deembed_movers": [],
    "deembed_figure": None,
    "deembed_has_effect": None,
    "deembed_state_changed": None,
    "vocab_contribution_present": False,
    "vocab_contribution_promoted": [],
    "vocab_contribution_suppressed": [],
    "vocab_contribution_movers": [],
    "vocab_contribution_figure": None,
    "vocab_contribution_has_effect": None,
}

PAYLOAD_KEYS: tuple[str, ...] = (
    "ok",
    "state",
    "message",
    "view",
    "ablation",
    *_DEFAULT_FIELDS,
)


def empty_payload(
    state: str,
    message: str,
    view: str = "baseline",
    ablation: Optional[dict] = None,
) -> dict:
    """Build a payload with every contract key present and empty."""
    payload: dict[str, Any] = {
        "ok": True,
        "state": state,
        "message": message,
        "view": view,
        "ablation": ablation,
    }
    for key, default in _DEFAULT_FIELDS.items():
        # Fresh lists keep callers from mutating a shared default.
        payload[key] = list(default) if isinstance(default, list) else default
    return payload


__all__ = ["PAYLOAD_KEYS", "empty_payload"]
