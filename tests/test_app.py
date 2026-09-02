import json
import re
from types import SimpleNamespace

import pytest
import tensorflow as tf

import app
from checkpoint import CONFIG_FILENAME, CheckpointError, save_checkpoint
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


def test_load_callback_loads_checkpoint_and_renders_details(tmp_path):
    config = make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())

    status, metadata, device, diagram, summary = app.load_model_callback(
        str(tmp_path), manager
    )

    assert status == "Model loaded successfully."
    assert str(tmp_path) in metadata
    assert f"{config.vocab_size:,}" in metadata
    assert "CPU" in device
    assert 'data-stage="embedding"' in diagram
    assert 'data-stage="vocabulary_projection"' in diagram
    assert "one_block_post_norm_causal_lm" in summary
    assert manager.loaded_state is not None
    assert manager.loaded_state.checkpoint.config == config


def test_load_callback_reports_missing_folder_and_has_no_model(tmp_path):
    manager = app.ModelManager(device_detector=lambda: fake_device())

    status, metadata, device, diagram, summary = app.load_model_callback(
        str(tmp_path / "missing"), manager
    )

    assert "Checkpoint could not be loaded" in status
    assert "does not exist" in status
    assert metadata == ""
    assert device.endswith("`not loaded`")
    assert summary == ""
    assert manager.loaded_state is None
    assert 'data-stage="embedding"' in diagram


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


def test_load_rejects_an_empty_folder_path():
    manager = app.ModelManager(device_detector=lambda: fake_device())

    result = manager.load("   ")

    assert not result.success
    assert "Enter a checkpoint folder path" in result.status
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


def test_analyze_without_model_reports_and_clears():
    manager = app.ModelManager(device_detector=lambda: fake_device())

    status, token_count, warning, token_rows, next_token_rows = (
        app.analyze_prompt_callback("hello", manager)
    )

    assert "Load a checkpoint" in status
    assert token_count == ""
    assert warning == ""
    assert token_rows == []
    assert next_token_rows == []


def test_analyze_callback_renders_token_and_prediction_tables(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))

    status, token_count, warning, token_rows, next_token_rows = (
        app.analyze_prompt_callback("hello , world", manager)
    )

    assert "Analysis complete" in status
    assert token_count == "**Processed tokens:** `3` of `6`"
    assert warning == ""
    assert [row[:3] for row in token_rows] == [
        [0, "hello", 2],
        [1, ",", 3],
        [2, "world", 4],
    ]
    assert len(next_token_rows) == 5
    assert all(len(row) == 4 for row in next_token_rows)
    assert all(re.fullmatch(r"\d\.\d{4}", row[3]) for row in next_token_rows)


def test_failed_analysis_clears_previous_results(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    status, token_count, warning, token_rows, next_token_rows = (
        app.analyze_prompt_callback("   ", manager)
    )

    assert "Enter a prompt first" in status
    assert token_count == ""
    assert warning == ""
    assert token_rows == []
    assert next_token_rows == []


def test_create_app_does_not_launch_server():
    demo = app.create_app(app.ModelManager(device_detector=lambda: fake_device()))

    assert isinstance(demo, app.gr.Blocks)
    assert app.launch_kwargs() == {
        "server_name": "127.0.0.1",
        "share": False,
    }


def test_analyze_and_inspect_callback_returns_capture_defaults(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))

    outputs = app.analyze_and_inspect_callback("hello , world", manager)

    assert len(outputs) == 11
    (status, token_count, warning, token_rows, next_rows,
     location, token, explanation, stats, plot, diagram) = outputs
    assert "Analysis complete" in status
    assert manager.inspection_session is not None
    assert manager.inspection_session.analysis.token_count == 3
    assert location["value"] == "output_norm"
    assert token["value"] == "2"
    assert "Final block output" in explanation
    assert "Residual Stream" in explanation
    assert "3 \u00d7 8" in stats
    assert isinstance(plot, app.Figure)
    assert 'class="ct-stage ct-selected" data-stage="output_norm"' in diagram
    assert 'class="ct-stage" data-stage="ffn_hidden"' in diagram


def test_selection_reuses_captured_data_without_running_the_model(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    class ExplodingModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("selection must not run the model")

    checkpoint = manager.loaded_state.checkpoint
    object.__setattr__(checkpoint, "model", ExplodingModel())

    explanation, stats, plot, diagram = app.select_location_callback(
        "ffn_hidden", "1", manager
    )

    assert "FFN hidden activation" in explanation
    assert "FFN" in explanation
    assert "3 \u00d7 8" in stats
    assert isinstance(plot, app.Figure)
    assert 'class="ct-stage ct-selected" data-stage="ffn_hidden"' in diagram


def test_selection_can_switch_back_to_the_default_location(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    explanation, _, _, diagram = app.select_location_callback(
        "output_norm", "0", manager
    )

    assert "Final block output" in explanation
    assert 'class="ct-stage ct-selected" data-stage="output_norm"' in diagram


def test_selection_before_analysis_reports_awaiting_state(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))

    explanation, stats, plot, diagram = app.select_location_callback(
        "output_norm", "0", manager
    )

    assert explanation == app.INSPECT_AWAITING
    assert stats == ""
    assert plot is None
    assert "ct-selected" not in diagram


def test_failed_analysis_clears_the_stored_capture(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    assert manager.inspection_session is not None

    outputs = app.analyze_and_inspect_callback("   ", manager)

    assert "Enter a prompt first" in outputs[0]
    assert manager.inspection_session is None
    assert outputs[5]["choices"] == []
    assert outputs[5]["value"] is None
    assert outputs[6]["choices"] == []
    assert outputs[7] == app.INSPECT_AWAITING
    assert outputs[8] == ""
    assert outputs[9] is None
    assert "ct-selected" not in outputs[10]


def test_loading_a_new_checkpoint_clears_the_stored_capture(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_checkpoint(first, seed=1)
    make_checkpoint(second, seed=2)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(first))
    app.analyze_prompt_callback("hello , world", manager)
    assert manager.inspection_session is not None

    manager.load(str(second))

    assert manager.inspection_session is None
