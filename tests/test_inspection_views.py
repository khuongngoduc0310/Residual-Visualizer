"""Tests for the inspection, de-embedding, and attribution views."""

import json

import numpy as np
import pytest
from support import (
    VOCABULARY,
    analyze_fixture,
    block0,
    model_manager,
    write_checkpoint,
)

import engine
import inspection_views
from inspection import (
    ABLATABLE_NODES,
    BRANCHES,
    DEFAULT_NODE_KEY,
    EMBEDDING_COMPONENTS,
    SPINE_NODES,
    STREAM_NODES,
    TRACE_ORDER,
    VOCAB_CONTRIBUTABLE_NODES,
    block_node_key,
)


def test_deembedding_projects_residual_state_through_output_matrix(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(
        manager, "output_norm", 1, "baseline", deembed=True
    )

    assert payload["node"]["deembeddable"] is True
    assert payload["deembed_present"] is True
    assert payload["deembed_figure"]["data"]

    probabilities = manager.inspection_session.analysis.capture.probabilities[1]
    assert payload["deembed_top"][0]["token_id"] == int(np.argmax(probabilities))
    assert payload["deembed_top"][0]["probability"] == pytest.approx(
        float(np.max(probabilities)),
        rel=1e-5,
    )
    expected_order = [
        int(token_id)
        for token_id in np.argsort(-probabilities)[: len(payload["deembed_top"])]
    ]
    assert [row["token_id"] for row in payload["deembed_top"]] == expected_order

    hidden = inspection_views.inspect_node_payload(
        manager, block0("ffn_hidden"), 1, "baseline", deembed=True
    )
    assert hidden["node"]["deembeddable"] is False
    assert hidden["deembed_present"] is False


def test_deembedding_normalizes_raw_residual_states_before_projection(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    raw = inspection_views.inspect_node_payload(
        manager,
        block_node_key(2, "ffn_residual"),
        1,
        "baseline",
        deembed=True,
    )
    normalized = inspection_views.inspect_node_payload(
        manager, "output_norm", 1, "baseline", deembed=True
    )

    assert raw["deembed_present"] is True
    assert [row["token_id"] for row in raw["deembed_top"]] == [
        row["token_id"] for row in normalized["deembed_top"]
    ]
    for raw_row, normalized_row in zip(
        raw["deembed_top"], normalized["deembed_top"], strict=False
    ):
        assert raw_row["probability"] == pytest.approx(
            normalized_row["probability"], rel=1e-5, abs=1e-7
        )


@pytest.mark.parametrize("stage", ["attention_update", "ffn_update"])
def test_update_vocabulary_contributions_use_final_norm_context(tmp_path, stage):
    manager, _ = analyze_fixture(tmp_path)
    node_key = block0(stage)
    position = 1

    payload = inspection_views.inspect_node_payload(
        manager, node_key, position, "baseline", vocab_contributions=True
    )

    checkpoint = manager.loaded_state.checkpoint
    locations = manager.inspection_session.analysis.capture.locations
    update = locations[node_key][position]
    final_residual = locations[block_node_key(2, "ffn_residual")][position]
    final_norm = checkpoint.model.get_layer("final_output_layer_norm")
    gamma = final_norm.get_weights()[0]
    kernel = checkpoint.model.get_layer("token_probabilities").get_weights()[0]
    expected = (
        (update - np.mean(update))
        / np.sqrt(
            np.mean(np.square(final_residual - np.mean(final_residual)))
            + final_norm.epsilon
        )
        * gamma
    ) @ kernel
    rows = {
        row["token_id"]: row["logit_contribution"]
        for row in (
            payload["vocab_contribution_promoted"]
            + payload["vocab_contribution_suppressed"]
        )
    }

    assert payload["node"]["vocab_contributable"] is True
    assert payload["node"]["deembeddable"] is False
    assert payload["vocab_contribution_present"] is True
    assert payload["deembed_present"] is False
    assert payload["vocab_contribution_figure"]["data"]
    assert set(rows) == set(range(checkpoint.config.vocab_size))
    for token_id, contribution in rows.items():
        assert contribution == pytest.approx(expected[token_id], rel=1e-5, abs=1e-7)
    assert all(
        row["logit_contribution"] > 0.0
        for row in payload["vocab_contribution_promoted"]
    )
    assert all(
        row["logit_contribution"] < 0.0
        for row in payload["vocab_contribution_suppressed"]
    )


def test_update_vocabulary_contributions_exclude_output_bias_and_softmax(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    def inspect():
        return inspection_views.inspect_node_payload(
            manager,
            block0("attention_update"),
            1,
            "baseline",
            vocab_contributions=True,
        )

    before = inspect()
    projection = manager.loaded_state.checkpoint.model.get_layer("token_probabilities")
    kernel, bias = projection.get_weights()
    projection.set_weights([kernel, bias + np.linspace(1.0, 3.0, len(bias))])

    after = inspect()

    def contribution_map(payload):
        return {
            row["token_id"]: row["logit_contribution"]
            for row in (
                payload["vocab_contribution_promoted"]
                + payload["vocab_contribution_suppressed"]
            )
        }

    assert contribution_map(after) == pytest.approx(contribution_map(before))
    values = [
        row["logit_contribution"]
        for row in (
            after["vocab_contribution_promoted"]
            + after["vocab_contribution_suppressed"]
        )
    ]
    assert any(value < 0.0 for value in values)
    assert sum(values) != pytest.approx(1.0)


def test_direct_attribution_scales_before_projecting_large_updates(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    checkpoint = manager.loaded_state.checkpoint
    width = checkpoint.config.embedding_dim
    magnitude = np.finfo(np.float32).max / 2.0
    update = np.array(
        [[magnitude if index % 2 == 0 else -magnitude for index in range(width)]],
        dtype=np.float32,
    )
    final_residual = update.copy()

    contributions = inspection_views.direct_logit_contributions(
        update, final_residual, 0, checkpoint
    )

    assert np.all(np.isfinite(contributions))


def test_direct_attributions_reconstruct_the_final_logits(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    checkpoint = manager.loaded_state.checkpoint
    locations = manager.inspection_session.analysis.capture.locations
    final_residual_key = block_node_key(2, "ffn_residual")
    position = 1
    component_keys = ["embedding", *VOCAB_CONTRIBUTABLE_NODES]
    contributions = sum(
        (
            inspection_views.direct_logit_contributions(
                locations[key],
                locations[final_residual_key],
                position,
                checkpoint,
            )
            for key in component_keys
        ),
        np.zeros(checkpoint.config.vocab_size),
    )
    final_norm = checkpoint.model.get_layer("final_output_layer_norm")
    _, beta = final_norm.get_weights()
    kernel, bias = checkpoint.model.get_layer("token_probabilities").get_weights()
    reconstructed = contributions + beta @ kernel + bias
    expected = locations["output_norm"][position] @ kernel + bias

    np.testing.assert_allclose(reconstructed, expected, rtol=1e-5, atol=1e-6)


def test_update_vocabulary_contributions_compare_ablated_capture(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    engine.ablate_feature_payload(
        manager,
        block0("ffn_hidden"),
        list(range(manager.loaded_state.checkpoint.config.feed_forward_dim)),
        "zero",
        "token",
        1,
    )

    payload = inspection_views.inspect_node_payload(
        manager, block0("ffn_update"), 1, "ablated", vocab_contributions=True
    )

    assert payload["vocab_contribution_present"] is True
    assert payload["vocab_contribution_has_effect"] is True
    assert payload["vocab_contribution_movers"]
    assert payload["vocab_contribution_figure"]["data"]
    checkpoint = manager.loaded_state.checkpoint
    baseline_locations = manager.inspection_session.analysis.capture.locations
    ablated_locations = manager.inspection_session.ablated.analysis.capture.locations
    final_residual_key = block_node_key(2, "ffn_residual")
    expected_baseline = inspection_views.direct_logit_contributions(
        baseline_locations[block0("ffn_update")],
        baseline_locations[final_residual_key],
        1,
        checkpoint,
    )
    expected_ablated = inspection_views.direct_logit_contributions(
        ablated_locations[block0("ffn_update")],
        ablated_locations[final_residual_key],
        1,
        checkpoint,
    )
    for row in payload["vocab_contribution_movers"]:
        token_id = row["token_id"]
        assert row["baseline_contribution"] == pytest.approx(
            expected_baseline[token_id], rel=1e-5, abs=1e-7
        )
        assert row["ablated_contribution"] == pytest.approx(
            expected_ablated[token_id], rel=1e-5, abs=1e-7
        )
        assert row["delta"] == pytest.approx(
            expected_ablated[token_id] - expected_baseline[token_id],
            rel=1e-5,
            abs=1e-7,
        )


def test_update_vocabulary_contributions_report_no_post_norm_effect(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    engine.ablate_feature_payload(manager, "output_norm", [0], "zero", "token", 1)

    payload = inspection_views.inspect_node_payload(
        manager, block0("attention_update"), 1, "ablated", vocab_contributions=True
    )

    assert payload["vocab_contribution_present"] is True
    assert payload["vocab_contribution_has_effect"] is False
    assert payload["vocab_contribution_movers"] == []
    assert payload["vocab_contribution_figure"] is None
    assert (
        payload["vocab_contribution_promoted"]
        or payload["vocab_contribution_suppressed"]
    )


def test_vocabulary_contribution_request_ignores_unsupported_nodes(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(
        manager, block0("ffn_hidden"), 1, "baseline", vocab_contributions=True
    )

    assert payload["node"]["vocab_contributable"] is False
    assert payload["vocab_contribution_present"] is False
    assert payload["vocab_contribution_promoted"] == []


def test_vocabulary_contribution_failure_returns_inspection_error(
    tmp_path, monkeypatch
):
    manager, _ = analyze_fixture(tmp_path)

    def fail(*_args, **_kwargs):
        raise inspection_views.InspectionError(
            "Vocabulary contributions were not finite."
        )

    monkeypatch.setattr(inspection_views, "populate_vocab_contribution_payload", fail)
    payload = inspection_views.inspect_node_payload(
        manager, block0("attention_update"), 1, "baseline", vocab_contributions=True
    )

    assert payload["state"] == "error"
    assert payload["message"] == "Vocabulary contributions were not finite."
    assert payload["vocab_contribution_present"] is False


def test_deembedding_compares_baseline_and_ablated_residual_states(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    engine.ablate_feature_payload(
        manager,
        block0("ffn_hidden"),
        list(range(manager.loaded_state.checkpoint.config.feed_forward_dim)),
        "zero",
        "token",
        1,
    )

    payload = inspection_views.inspect_node_payload(
        manager, block0("ffn_residual"), 1, "ablated", deembed=True
    )

    assert payload["deembed_present"] is True
    assert payload["deembed_movers"]
    assert payload["deembed_figure"]["data"]


def test_deembedding_same_ablated_residual_keeps_projected_predictions(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    position = 1
    checkpoint = manager.loaded_state.checkpoint
    baseline_values = manager.inspection_session.analysis.capture.locations[
        block0("ffn_residual")
    ]
    projection = checkpoint.model.get_layer("token_probabilities")
    kernel = projection.get_weights()[0]
    scores = np.abs(baseline_values[position]) * np.ptp(kernel, axis=1)
    dimension = int(np.argmax(scores))
    assert scores[dimension] > 1e-12

    result = engine.ablate_feature_payload(
        manager,
        block0("ffn_residual"),
        [dimension],
        "zero",
        "token",
        position,
    )
    payload = inspection_views.inspect_node_payload(
        manager, block0("ffn_residual"), position, "ablated", deembed=True
    )

    assert result["ok"]
    assert (
        manager.inspection_session.ablated.analysis.capture.locations[
            block0("ffn_residual")
        ][position, dimension]
        == 0.0
    )
    assert payload["deembed_present"] is True
    assert payload["deembed_state_changed"] is True
    assert payload["deembed_has_effect"] is True
    assert payload["deembed_top"]
    assert payload["deembed_movers"]
    assert payload["deembed_figure"]["data"]
    assert all(np.isfinite(row["probability"]) for row in payload["deembed_top"])
    assert any(abs(row["delta"]) > 1e-12 for row in payload["deembed_movers"])


def test_non_baseline_view_requires_an_active_ablation(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(manager, "output_norm", 0, "diff")

    assert payload["state"] == "error"
    assert "No ablation is active" in payload["message"]


def test_inspect_before_analysis_reports_awaiting_state(tmp_path):
    write_checkpoint(tmp_path)
    manager = model_manager()
    manager.load(str(tmp_path))

    payload = inspection_views.inspect_node_payload(manager)

    assert payload["ok"]
    assert payload["state"] == "awaiting"
    assert payload["message"] == inspection_views.INSPECT_AWAITING
    assert payload["node"] is None
    assert payload["selected_position"] is None
    assert payload["token_choices"] == []
    assert payload["shape"] is None
    assert payload["capture"] is None
    assert payload["scale"] is None
    assert payload["tile"] is None
    assert payload["map_figure"] is None
    assert payload["pattern_figure"] is None
    assert payload["readout_figure"] is None
    assert payload["entropy_figure"] is None
    assert payload["readout_rows"] == []
    assert payload["vocab_contribution_present"] is False
    assert payload["vocab_contribution_promoted"] == []
    assert payload["vocab_contribution_suppressed"] == []
    assert payload["vocab_contribution_movers"] == []
    assert payload["vocab_contribution_figure"] is None


def test_inspect_returns_capture_defaults(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(manager)

    assert payload["ok"]
    assert payload["state"] == "ready"
    assert payload["node"]["key"] == DEFAULT_NODE_KEY
    assert payload["node"]["label"] == "Layer norm - readout input"
    assert payload["node"]["family"] == "norm"
    assert payload["node"]["normalized"] is True
    assert "final model-level normalization" in payload["node"]["explanation"]
    assert payload["node"]["prev_key"] == block_node_key(2, "ffn_residual")
    assert payload["node"]["next_key"] == "readout"
    assert payload["selected_position"] == 2
    assert payload["token_choices"] == [
        {"position": 0, "text": "hello"},
        {"position": 1, "text": ","},
        {"position": 2, "text": "world"},
    ]
    assert payload["shape"] == {"seq_len": 3, "width": 8}
    assert set(payload["capture"]) == {"min", "mean", "max"}
    assert payload["capture"]["max"] >= payload["capture"]["min"]
    assert payload["scale"] == {
        "lower": -payload["scale"]["upper"],
        "upper": payload["scale"]["upper"],
    }
    assert payload["scale"]["upper"] >= 0
    assert payload["tile"] == {"rows": 2, "cols": 4}
    assert "data" in payload["map_figure"]
    assert "layout" in payload["map_figure"]
    assert len(payload["map_figure"]["data"]) == 1


def test_inspect_uses_stored_data_without_running_the_model(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    class ExplodingModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("selection must not run the model")

    checkpoint = manager.loaded_state.checkpoint
    object.__setattr__(checkpoint, "model", ExplodingModel())

    payload = inspection_views.inspect_node_payload(manager, block0("ffn_hidden"), 1)

    assert payload["state"] == "ready"
    assert payload["node"]["key"] == block0("ffn_hidden")
    assert payload["node"]["family"] == "hidden"
    assert payload["figure_kind"] == "hidden"
    assert payload["selected_position"] == 1
    assert set(payload["scale"]) == {"lower", "upper"}
    assert payload["scale"]["lower"] == 0.0
    assert payload["scale"]["upper"] > 0.0
    assert payload["map_figure"]["data"]
    assert payload["tile"] == {"rows": 2, "cols": 4}


def test_inspect_can_switch_back_to_the_default_node(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(manager, "output_norm", 0)

    assert payload["node"]["key"] == "output_norm"
    assert payload["node"]["label"] == "Layer norm - readout input"
    assert payload["selected_position"] == 0


def test_inspect_clamps_token_position_to_the_capture(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(manager, "output_norm", 999)
    high = inspection_views.inspect_node_payload(manager, "output_norm", -3)

    assert payload["selected_position"] == 2
    assert high["selected_position"] == 0


def test_inspect_reports_an_unknown_node(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(manager, "not_a_node", 0)

    assert payload["state"] == "error"
    assert "Unknown stream node" in payload["message"]


def test_each_node_normalizes_its_own_color_scale(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    embedding = inspection_views.inspect_node_payload(manager, "embedding", None)
    after_attention = inspection_views.inspect_node_payload(
        manager, block0("attention_residual"), None
    )

    for payload in (embedding, after_attention):
        assert set(payload["scale"]) == {"lower", "upper"}
        assert payload["scale"]["lower"] == -payload["scale"]["upper"]
    assert embedding["scale"]["upper"] != after_attention["scale"]["upper"]


def test_attention_pattern_node_returns_pattern_view(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = inspection_views.inspect_node_payload(
        manager, block0("attention_pattern"), 1
    )

    assert payload["state"] == "ready"
    assert payload["node"]["kind"] == "pattern"
    assert payload["figure_kind"] == "pattern"
    assert payload["pattern_figure"]["data"]
    assert payload["map_figure"] is None
    assert payload["tile"] is None
    assert payload["shape"] == {"seq_len": 3, "width": 3}


def test_embedding_components_node_views(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    for key in EMBEDDING_COMPONENTS:
        payload = inspection_views.inspect_node_payload(manager, key, 0)
        assert payload["state"] == "ready"
        assert payload["node"]["kind"] == "component"
        assert payload["node"]["family"] == "components"
        assert payload["shape"] == {"seq_len": 3, "width": 8}
        assert payload["tile"] == {"rows": 2, "cols": 4}
        assert payload["map_figure"]["data"]
        assert payload["map_figure"]["data"][0]["zmin"] == payload["scale"]["lower"]


def test_readout_node_returns_topk_rows_and_entropy(tmp_path):
    manager, payload = analyze_fixture(tmp_path)
    analysis = manager.inspection_session.analysis

    inspected = inspection_views.inspect_node_payload(manager, "readout", 2)

    assert inspected["state"] == "ready"
    assert inspected["node"]["kind"] == "readout"
    assert inspected["figure_kind"] == "readout_topk"
    assert inspected["readout_figure"]["data"]
    assert inspected["entropy_figure"]["data"]
    assert inspected["shape"]["width"] == len(VOCABULARY)
    assert {row["rank"] for row in inspected["readout_rows"]} == set(range(1, 7))
    probabilities = [row["probability"] for row in inspected["readout_rows"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert probabilities[0] == max(analysis.capture.probabilities[2].tolist())
    assert inspected["capture"]["max"] >= inspected["capture"]["min"]


def test_every_node_renders_a_view_after_analysis(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    keys = [node.key for node in STREAM_NODES]

    for node in STREAM_NODES:
        payload = inspection_views.inspect_node_payload(manager, node.key, 0)
        assert payload["state"] == "ready", node.key
        assert payload["node"]["trace_index"] == keys.index(node.key)
        assert payload["selected_position"] == 0


def test_load_payloads_are_json_serializable(tmp_path):
    write_checkpoint(tmp_path)
    manager = model_manager()
    loaded = engine.load_model_payload(manager, str(tmp_path))
    engine.analyze_prompt_payload(manager, "hello , world")
    activation = inspection_views.inspect_node_payload(manager, block0("ffn_update"), 1)
    pattern = inspection_views.inspect_node_payload(
        manager, block0("attention_pattern"), 1
    )
    readout = inspection_views.inspect_node_payload(manager, "readout", 2)
    contribution = inspection_views.inspect_node_payload(
        manager, block0("attention_update"), 1, "baseline", vocab_contributions=True
    )

    for payload in (loaded, activation, pattern, readout, contribution):
        json.loads(json.dumps(payload, allow_nan=False))


def test_options_payload_describes_the_stream_graph():
    payload = inspection_views.options_payload()

    graph = payload["graph"]
    assert graph["default_node"] == DEFAULT_NODE_KEY
    keys = [node["key"] for node in graph["nodes"]]
    assert keys == list(TRACE_ORDER)
    assert graph["spine"] == list(SPINE_NODES)
    assert len(graph["spine_links"]) == len(graph["spine"])
    assert graph["trace"] == list(TRACE_ORDER)
    assert [branch["key"] for branch in graph["branches"]] == [
        branch.key for branch in BRANCHES
    ]
    assert len(graph["branches"]) == 6
    attention = graph["branches"][0]
    assert attention["reads"] == "embedding"
    assert attention["adds_before"] == block0("attention_residual")
    assert attention["path"] == [
        block0("attention_input_norm"),
        block0("attention_update"),
    ]
    assert attention["observables"] == [block0("attention_pattern")]
    assert attention["kind"] == "attention"
    assert attention["block_index"] == 0
    assert attention["side"] == "above"
    ffn = graph["branches"][1]
    assert ffn["reads"] == block0("attention_residual")
    assert ffn["path"] == [
        block0("ffn_input_norm"),
        block0("ffn_hidden"),
        block0("ffn_update"),
    ]
    assert ffn["observables"] == []
    assert graph["branches"][-1]["block_index"] == 2
    assert graph["components"] == list(EMBEDDING_COMPONENTS)
    assert {
        node["key"] for node in graph["nodes"] if node["vocab_contributable"]
    } == set(VOCAB_CONTRIBUTABLE_NODES)
    assert [node["key"] for node in payload["ablation_nodes"]] == list(ABLATABLE_NODES)
    output_norm = next(node for node in graph["nodes"] if node["key"] == "output_norm")
    assert output_norm["normalized"] is True
    assert output_norm["next_key"] == "readout"
    block_two_hidden = next(
        node
        for node in graph["nodes"]
        if node["key"] == block_node_key(1, "ffn_hidden")
    )
    assert block_two_hidden["block_index"] == 1
    assert block_two_hidden["stage"] == "ffn_hidden"
    assert block_two_hidden["width_source"] == "ffn"
    assert payload["locations"][0]["key"] == "token_embeddings"
