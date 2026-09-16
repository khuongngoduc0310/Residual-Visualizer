"""The inspection payload key set is a contract shared with the frontend."""

from support import analyze_fixture, block0

import inspection_views
from inspection_payload import PAYLOAD_KEYS, empty_payload

EXPECTED_KEYS = (
    "ok",
    "state",
    "message",
    "view",
    "ablation",
    "node",
    "selected_position",
    "token_choices",
    "shape",
    "capture",
    "scale",
    "tile",
    "figure_kind",
    "map_figure",
    "pattern_figure",
    "readout_figure",
    "entropy_figure",
    "readout_rows",
    "readout_compare",
    "readout_compare_figure",
    "position_effects",
    "deembed_present",
    "deembed_top",
    "deembed_movers",
    "deembed_figure",
    "deembed_has_effect",
    "deembed_state_changed",
    "vocab_contribution_present",
    "vocab_contribution_promoted",
    "vocab_contribution_suppressed",
    "vocab_contribution_movers",
    "vocab_contribution_figure",
    "vocab_contribution_has_effect",
)


def test_payload_keys_match_the_documented_contract():
    assert PAYLOAD_KEYS == EXPECTED_KEYS


def test_empty_payload_has_every_contract_key_in_order():
    payload = empty_payload("ready", "")

    assert tuple(payload) == EXPECTED_KEYS
    assert payload["ok"] is True


def test_empty_payload_passes_through_state_view_and_ablation():
    ablation = {"node_key": "output_norm"}

    payload = empty_payload("error", "boom", view="diff", ablation=ablation)

    assert payload["state"] == "error"
    assert payload["message"] == "boom"
    assert payload["view"] == "diff"
    assert payload["ablation"] is ablation


def test_empty_payload_uses_fresh_lists_per_call():
    first = empty_payload("ready", "")
    second = empty_payload("ready", "")

    first["token_choices"].append({"position": 0})
    first["deembed_top"].append({"rank": 1})

    assert second["token_choices"] == []
    assert second["deembed_top"] == []


def test_inspect_endpoint_returns_exactly_the_contract_keys(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payloads = [
        inspection_views.inspect_node_payload(manager),
        inspection_views.inspect_node_payload(
            manager, "output_norm", 1, "baseline", deembed=True
        ),
        inspection_views.inspect_node_payload(
            manager,
            block0("attention_update"),
            1,
            "baseline",
            vocab_contributions=True,
        ),
        inspection_views.inspect_node_payload(manager, "readout", 1),
        inspection_views.inspect_node_payload(manager, "not_a_node"),
    ]

    for payload in payloads:
        assert tuple(payload) == EXPECTED_KEYS
