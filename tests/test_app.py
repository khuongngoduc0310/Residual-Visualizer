import json
from types import SimpleNamespace

import pytest
import tensorflow as tf

import app
from checkpoint import CONFIG_FILENAME, CheckpointError, save_checkpoint
from inspection import (
    DEFAULT_NODE_KEY,
    EMBEDDING_COMPONENTS,
    STREAM_NODES,
    TRACE_ORDER,
)
from model import ModelConfig, build_model


VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]


def tiny_config(**changes):
    values = {
        "vocab_size": len(VOCABULARY),
        "max_len": 6,
        "embedding_dim": 8,
        "num_heads": 2,
        "key_dim": 4,
        "feed_forward_dim": 8,
        "dropout_rate": 0.0,
    }
    values.update(changes)
    return ModelConfig(**values)


def make_checkpoint(path, seed=9):
    tf.keras.utils.set_random_seed(seed)
    config = tiny_config()
    model = build_model(config)
    save_checkpoint(path, model, VOCABULARY, config)
    return config


def fake_device(label="CPU", tf_device="/CPU:0", is_gpu=False):
    return app.ComputeDevice(label=label, tf_device=tf_device, is_gpu=is_gpu)


def loaded_manager(path, seed=9):
    make_checkpoint(path, seed=seed)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(path))
    return manager


def analyze_fixture(path, prompt="hello , world"):
    manager = loaded_manager(path)
    payload = app.analyze_prompt_payload(manager, prompt)
    assert payload["ok"]
    return manager, payload


def node_keys():
    return [node.key for node in STREAM_NODES]


# --------------------------------------------------------------------------- #
# Checkpoint loading payloads


def test_load_payload_loads_checkpoint_and_describes_the_model(tmp_path):
    config = make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())

    payload = app.load_model_payload(manager, str(tmp_path))

    assert payload["ok"]
    assert payload["loaded"]
    assert payload["status"] == "Model loaded successfully."
    assert payload["meta"]["path"] == str(tmp_path)
    assert payload["meta"]["architecture"] == app.ARCHITECTURE_NAME
    assert payload["meta"]["vocab_size"] == config.vocab_size
    assert payload["meta"]["max_len"] == config.max_len
    assert payload["meta"]["embedding_dim"] == config.embedding_dim
    assert payload["meta"]["num_heads"] == config.num_heads
    assert payload["meta"]["key_dim"] == config.key_dim
    assert payload["meta"]["feed_forward_dim"] == config.feed_forward_dim
    assert payload["device_label"] == "CPU"
    assert "one_block_post_norm_causal_lm" in payload["summary"]
    assert manager.loaded_state is not None
    assert manager.loaded_state.checkpoint.config == config


def test_load_payload_reports_missing_folder_and_has_no_model(tmp_path):
    manager = app.ModelManager(device_detector=lambda: fake_device())

    payload = app.load_model_payload(manager, str(tmp_path / "missing"))

    assert not payload["ok"]
    assert "Checkpoint could not be loaded" in payload["status"]
    assert "does not exist" in payload["status"]
    assert not payload["loaded"]
    assert payload["meta"]["path"] is None
    assert payload["device_label"] is None
    assert payload["summary"] is None
    assert manager.loaded_state is None


def test_load_payload_rejects_an_empty_folder_path():
    manager = app.ModelManager(device_detector=lambda: fake_device())

    payload = app.load_model_payload(manager, "   ")

    assert not payload["ok"]
    assert "Enter a checkpoint folder path" in payload["status"]
    assert manager.loaded_state is None


def test_failed_replacement_unloads_previous_model(tmp_path):
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    make_checkpoint(valid_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())

    first = manager.load(str(valid_path))
    second = manager.load(str(tmp_path / "invalid"))

    assert first.success
    assert not second.success
    assert manager.loaded_state is None


def test_loading_a_second_checkpoint_releases_the_first(tmp_path):
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    make_checkpoint(first_path, seed=1)
    make_checkpoint(second_path, seed=2)
    clear_calls = []
    collect_calls = []
    manager = app.ModelManager(
        device_detector=lambda: fake_device(),
        session_clearer=lambda: clear_calls.append(True),
        collector=lambda: collect_calls.append(True),
    )

    manager.load(str(first_path))
    first_model = manager.loaded_state.checkpoint.model
    manager.load(str(second_path))

    assert manager.loaded_state.checkpoint_path == second_path
    assert manager.loaded_state.checkpoint.model is not first_model
    assert len(clear_calls) == 2
    assert len(collect_calls) == 2


def test_malformed_config_is_reported_without_traceback(tmp_path):
    make_checkpoint(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
    manager = app.ModelManager(device_detector=lambda: fake_device())

    result = manager.load(str(tmp_path))

    assert not result.success
    assert "Checkpoint could not be loaded" in result.status
    assert "JSONDecodeError" not in result.status
    assert manager.loaded_state is None


def test_mismatched_checkpoint_is_reported_and_clears_previous_model(tmp_path):
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    make_checkpoint(valid_path)
    config_path = valid_path / CONFIG_FILENAME
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["model"]["feed_forward_dim"] = 12
    config_path.write_text(json.dumps(document), encoding="utf-8")
    manager = app.ModelManager(device_detector=lambda: fake_device())

    result = manager.load(str(valid_path))

    assert not result.success
    assert "do not match" in result.status
    assert manager.loaded_state is None


def test_use_loaded_state_requires_and_yields_the_active_model(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())

    with pytest.raises(CheckpointError, match="Load a checkpoint"):
        with manager.use_loaded_state():
            pass

    manager.load(str(tmp_path))
    with manager.use_loaded_state() as state:
        assert state.checkpoint_path == tmp_path


def test_cuda_device_requires_cuda_build_and_visible_gpu(monkeypatch):
    monkeypatch.setattr(
        app.tf.sysconfig,
        "get_build_info",
        lambda: {"is_cuda_build": True},
    )
    monkeypatch.setattr(app.tf.test, "is_built_with_cuda", lambda: True)
    monkeypatch.setattr(
        app.tf.config,
        "list_physical_devices",
        lambda kind: [SimpleNamespace(name="GPU:0")] if kind == "GPU" else [],
    )
    monkeypatch.setattr(
        app.tf.config.experimental,
        "get_device_details",
        lambda device: {"device_name": "Test NVIDIA GPU"},
    )

    device = app.detect_compute_device()

    assert device.is_gpu
    assert device.tf_device == "/GPU:0"
    assert device.label == "CUDA GPU: Test NVIDIA GPU"


def test_cpu_device_does_not_claim_cuda(monkeypatch):
    monkeypatch.setattr(
        app.tf.sysconfig,
        "get_build_info",
        lambda: {"is_cuda_build": False},
    )
    monkeypatch.setattr(app.tf.test, "is_built_with_cuda", lambda: False)
    monkeypatch.setattr(app.tf.config, "list_physical_devices", lambda kind: [])

    device = app.detect_compute_device()

    assert not device.is_gpu
    assert device.tf_device == "/CPU:0"
    assert device.label == "CPU"


# --------------------------------------------------------------------------- #
# Prompt analysis payloads


def test_analyze_without_model_reports_and_clears():
    manager = app.ModelManager(device_detector=lambda: fake_device())

    payload = app.analyze_prompt_payload(manager, "hello")

    assert not payload["ok"]
    assert "Load a checkpoint" in payload["status"]
    assert payload["token_count"] is None
    assert payload["tokens"] == []
    assert payload["next_tokens"] == []


def test_analyze_payload_renders_token_and_prediction_tables(tmp_path):
    manager = loaded_manager(tmp_path)

    payload = app.analyze_prompt_payload(manager, "hello , world")

    assert payload["ok"]
    assert payload["status"] == "Analysis complete for 3 processed token(s)."
    assert payload["token_count"] == 3
    assert payload["max_len"] == 6
    assert payload["unknown_count"] == 0
    assert payload["tokens"] == [
        {"position": 0, "text": "hello", "token_id": 2},
        {"position": 1, "text": ",", "token_id": 3},
        {"position": 2, "text": "world", "token_id": 4},
    ]
    assert len(payload["next_tokens"]) == 5
    for token in payload["next_tokens"]:
        assert set(token) == {"rank", "text", "token_id", "probability"}
        assert isinstance(token["probability"], float)
    assert [t["rank"] for t in payload["next_tokens"]] == [1, 2, 3, 4, 5]
    assert manager.inspection_session is not None
    assert manager.inspection_session.analysis.token_count == 3


def test_failed_analysis_clears_previous_results(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_payload(manager, "hello , world")
    assert manager.inspection_session is not None

    payload = app.analyze_prompt_payload(manager, "   ")

    assert not payload["ok"]
    assert payload["status"] == "Enter a prompt first."
    assert payload["token_count"] is None
    assert payload["tokens"] == []
    assert payload["next_tokens"] == []
    assert manager.inspection_session is None


# --------------------------------------------------------------------------- #
# Inspection payloads


def test_inspect_before_analysis_reports_awaiting_state(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))

    payload = app.inspect_node_payload(manager)

    assert payload["ok"]
    assert payload["state"] == "awaiting"
    assert payload["message"] == app.INSPECT_AWAITING
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


def test_inspect_returns_capture_defaults(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = app.inspect_node_payload(manager)

    assert payload["ok"]
    assert payload["state"] == "ready"
    assert payload["node"]["key"] == DEFAULT_NODE_KEY
    assert payload["node"]["label"] == "Layer norm \u00b7 block output"
    assert payload["node"]["family"] == "stream_norm"
    assert payload["node"]["normalized"] is True
    assert "layer-normalized block output" in payload["node"]["explanation"]
    assert payload["node"]["prev_key"] == "ffn_residual"
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

    payload = app.inspect_node_payload(manager, "ffn_hidden", 1)

    assert payload["state"] == "ready"
    assert payload["node"]["key"] == "ffn_hidden"
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

    payload = app.inspect_node_payload(manager, "output_norm", 0)

    assert payload["node"]["key"] == "output_norm"
    assert payload["node"]["label"] == "Layer norm \u00b7 block output"
    assert payload["selected_position"] == 0


def test_inspect_clamps_token_position_to_the_capture(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = app.inspect_node_payload(manager, "output_norm", 999)
    high = app.inspect_node_payload(manager, "output_norm", -3)

    assert payload["selected_position"] == 2
    assert high["selected_position"] == 0


def test_inspect_reports_an_unknown_node(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = app.inspect_node_payload(manager, "not_a_node", 0)

    assert payload["state"] == "error"
    assert "Unknown stream node" in payload["message"]


def test_each_node_normalizes_its_own_color_scale(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    embedding = app.inspect_node_payload(manager, "embedding", None)
    after_attention = app.inspect_node_payload(
        manager, "attention_residual", None
    )

    for payload in (embedding, after_attention):
        assert set(payload["scale"]) == {"lower", "upper"}
        assert payload["scale"]["lower"] == -payload["scale"]["upper"]
    assert embedding["scale"]["upper"] != after_attention["scale"]["upper"]


def test_attention_pattern_node_returns_pattern_view(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    payload = app.inspect_node_payload(manager, "attention_pattern", 1)

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
        payload = app.inspect_node_payload(manager, key, 0)
        assert payload["state"] == "ready"
        assert payload["node"]["kind"] == "component"
        assert payload["node"]["family"] == "components"
        assert payload["shape"] == {"seq_len": 3, "width": 8}
        assert payload["tile"] == {"rows": 2, "cols": 4}
        assert payload["map_figure"]["data"]
        assert payload["map_figure"]["data"][0]["zmin"] == \
            payload["scale"]["lower"]


def test_readout_node_returns_topk_rows_and_entropy(tmp_path):
    manager, payload = analyze_fixture(tmp_path)
    analysis = manager.inspection_session.analysis

    inspected = app.inspect_node_payload(manager, "readout", 2)

    assert inspected["state"] == "ready"
    assert inspected["node"]["kind"] == "readout"
    assert inspected["figure_kind"] == "readout_topk"
    assert inspected["readout_figure"]["data"]
    assert inspected["entropy_figure"]["data"]
    assert inspected["shape"]["width"] == len(VOCABULARY)
    assert {row["rank"] for row in inspected["readout_rows"]} == set(range(1, 7))
    probabilities = [row["probability"] for row in inspected["readout_rows"]]
    assert probabilities == sorted(probabilities, reverse=True)
    assert probabilities[0] == max(
        analysis.capture.probabilities[2].tolist()
    )
    assert inspected["capture"]["max"] >= inspected["capture"]["min"]


def test_every_node_renders_a_view_after_analysis(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    for node in STREAM_NODES:
        payload = app.inspect_node_payload(manager, node.key, 0)
        assert payload["state"] == "ready", node.key
        assert payload["node"]["trace_index"] == node_keys().index(node.key)
        assert payload["selected_position"] == 0


def test_load_payloads_are_json_serializable(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    loaded = app.load_model_payload(manager, str(tmp_path))
    app.analyze_prompt_payload(manager, "hello , world")
    activation = app.inspect_node_payload(manager, "ffn_update", 1)
    pattern = app.inspect_node_payload(manager, "attention_pattern", 1)
    readout = app.inspect_node_payload(manager, "readout", 2)

    for payload in (loaded, activation, pattern, readout):
        json.loads(json.dumps(payload))


def test_loading_a_new_checkpoint_clears_the_stored_capture(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_checkpoint(first, seed=1)
    make_checkpoint(second, seed=2)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(first))
    app.analyze_prompt_payload(manager, "hello , world")
    assert manager.inspection_session is not None

    manager.load(str(second))

    assert manager.inspection_session is None


# --------------------------------------------------------------------------- #
# Static options and app construction


def test_options_payload_describes_the_stream_graph():
    payload = app.options_payload()

    graph = payload["graph"]
    assert graph["default_node"] == DEFAULT_NODE_KEY
    keys = [node["key"] for node in graph["nodes"]]
    assert keys == list(TRACE_ORDER)
    assert graph["spine"] == [
        "embedding",
        "attention_residual",
        "attention_norm",
        "ffn_residual",
        "output_norm",
    ]
    assert len(graph["spine_links"]) == len(graph["spine"])
    assert graph["trace"] == list(TRACE_ORDER)
    assert [branch["key"] for branch in graph["branches"]] == [
        "attention",
        "ffn",
    ]
    attention = graph["branches"][0]
    assert attention["reads"] == "embedding"
    assert attention["adds_before"] == "attention_residual"
    assert attention["nodes"] == ["attention_pattern", "attention_update"]
    assert graph["components"] == list(EMBEDDING_COMPONENTS)
    output_norm = next(
        node for node in graph["nodes"] if node["key"] == "output_norm"
    )
    assert output_norm["normalized"] is True
    assert output_norm["next_key"] == "readout"
    assert payload["locations"][0]["key"] == "token_embeddings"


def test_create_app_builds_endpoints_without_launching():
    demo = app.create_app(app.ModelManager(device_detector=lambda: fake_device()))

    assert isinstance(demo, app.gr.Blocks)
    api_names = {fn.api_name for fn in demo.fns.values()}
    assert {
        "load_checkpoint",
        "analyze_prompt",
        "inspect_node",
        "options",
    } <= api_names
