import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
import plotly.graph_objects as go
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


def make_checkpoint(path, seed=9, **config_changes):
    tf.keras.utils.set_random_seed(seed)
    config = tiny_config(**config_changes)
    model = build_model(config)
    save_checkpoint(path, model, VOCABULARY, config)
    return config


def fake_device(label="CPU", tf_device="/CPU:0", is_gpu=False):
    return app.ComputeDevice(label=label, tf_device=tf_device, is_gpu=is_gpu)


def plot_value(output):
    return output.get("value") if isinstance(output, dict) else output


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


def test_failed_replacement_preserves_previous_model(tmp_path):
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    make_checkpoint(valid_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())

    first = manager.load(str(valid_path))
    second = manager.load(str(tmp_path / "invalid"))

    assert first.success
    assert not second.success
    assert manager.loaded_state is not None
    assert "previous model remains active" in second.status


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


def test_failed_analysis_preserves_previous_results(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    previous = app.analyze_prompt_callback("hello , world", manager)
    previous_session = manager.inspection_session

    status, token_count, warning, token_rows, next_token_rows = (
        app.analyze_prompt_callback("   ", manager)
    )

    assert "Enter a prompt first" in status
    assert (token_count, warning, token_rows, next_token_rows) == previous[1:]
    assert manager.inspection_session is previous_session


def test_create_app_does_not_launch_server():
    demo = app.create_app(app.ModelManager(device_detector=lambda: fake_device()))

    assert isinstance(demo, app.gr.Blocks)
    assert "background: #ffffff" in app.APP_CSS
    assert "ct-expand-visuals" in app.CLICK_BRIDGE_JS
    assert "ct-close-visuals" in app.CLICK_BRIDGE_JS
    assert "ct-visual-expanded" in app.APP_CSS
    assert app.launch_kwargs() == {
        "server_name": "127.0.0.1",
        "share": False,
        "theme": app.LIGHT_THEME,
        "run_history": False,
    }


def test_analyze_and_inspect_callback_returns_capture_defaults(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))

    outputs = app.analyze_and_inspect_callback("hello , world", manager)

    assert len(outputs) == 24
    (status, details, token_count, warning, token_rows, next_rows,
     location, token, normalization, mode, clipped, explanation, stats,
     visible_range, magnitude_plot, heatmap_plot, distribution_plot, diagram,
     comparison_a, comparison_b, delta, comparison_status, diagnostics,
     token_click) = outputs
    assert "Analysis complete" in status
    assert details == ""
    assert manager.inspection_session is not None
    assert manager.inspection_session.analysis.token_count == 3
    assert location["value"] == "output_norm"
    assert token["value"] == []
    assert normalization["value"] == "location"
    assert normalization["interactive"]
    assert normalization["choices"] == app.EMPTY_SELECTION_SCOPE_CHOICES
    assert mode["value"] == "square"
    assert clipped is False
    heatmap_plot = plot_value(heatmap_plot)
    assert isinstance(heatmap_plot.data[0], go.Heatmap)
    assert heatmap_plot.data[0].z.shape == (3, 8)
    values = manager.inspection_session.analysis.capture.locations["output_norm"]
    assert (heatmap_plot.data[0].zmin, heatmap_plot.data[0].zmax) == app.display_bounds(values)
    assert "Final block output" in explanation
    assert "Residual Stream" in explanation
    assert "Overview" in stats
    assert "Visible heatmap range" in visible_range
    assert magnitude_plot is None
    assert isinstance(heatmap_plot, go.Figure)
    assert distribution_plot is None
    assert 'class="ct-stage ct-selected" data-stage="output_norm"' in diagram
    assert 'class="ct-stage" data-stage="ffn_hidden"' in diagram
    assert comparison_status == "No comparison pinned."
    assert all(plot_value(value) is None for value in (comparison_a, comparison_b, delta))
    assert diagnostics["capture"]["available"]
    assert token_click == ""


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

    _, explanation, stats, visible_range, magnitude, heatmap, distribution, diagram, *_ = app.select_location_callback(
        "ffn_hidden", ["1"], False, "selected_point", manager
    )

    assert "FFN hidden activation" in explanation
    assert "FFN" in explanation
    assert "3 \u00d7 8" in stats
    assert "Visible heatmap range" in visible_range
    assert isinstance(magnitude, go.Figure)
    assert isinstance(plot_value(heatmap), go.Figure)
    assert isinstance(distribution, go.Figure)
    assert 'class="ct-stage ct-selected" data-stage="ffn_hidden"' in diagram


def test_selection_can_switch_back_to_the_default_location(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    _, explanation, _, _, _, heatmap, _, diagram, *_ = app.select_location_callback(
        "output_norm", ["0"], False, "selected_point", manager
    )

    assert "Final block output" in explanation
    assert isinstance(plot_value(heatmap), go.Figure)
    assert 'class="ct-stage ct-selected" data-stage="output_norm"' in diagram


def test_clicked_token_rerenders_all_views_and_updates_selector(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    token, explanation, stats, visible_range, magnitude, heatmap, distribution, *_ = (
        app.select_clicked_token_callback("1", "output_norm", ["2"], False, "selected_point", manager)
    )

    assert token["value"] == ["1"]
    assert "Final block output" in explanation
    assert "Selected token" in stats
    assert "Visible heatmap range" in visible_range
    assert all(isinstance(plot, go.Figure) for plot in (magnitude, plot_value(heatmap), distribution))


def test_overview_click_toggles_selection_and_drives_detail_overview_transitions(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello hello world", manager)
    payload = json.dumps({
        "schema": "circuit-tracer.click.v1",
        "view": "overview",
        "token_position": 1,
        "dimension": 3,
    })

    selected = app.activation_click_callback(
        payload, "output_norm", [], False, "location", manager
    )
    cleared = app.activation_click_callback(
        payload, "output_norm", ["1"], False, "location", manager
    )

    assert selected[0]["value"] == ["1"]
    assert plot_value(selected[5]).data[0].customdata[0, 0, 1] == "detail"
    assert selected[12]["choices"] == app.SCALE_SCOPE_CHOICES
    assert cleared[0]["value"] == []
    assert plot_value(cleared[5]).data[0].z.shape == (3, 8)
    assert plot_value(cleared[5]).data[0].customdata[0, 0, 1] == "overview"
    assert cleared[12]["choices"] == app.EMPTY_SELECTION_SCOPE_CHOICES
    assert cleared[12]["value"] == "location"


def test_token_selection_persists_by_position_across_location_navigation(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello hello world", manager)

    outputs = app.select_bridged_location_callback(
        "attention_update", ["0", "2"], False, "location", manager, "indexed"
    )

    assert outputs[1]["value"] == ["0", "2"]
    np.testing.assert_array_equal(plot_value(outputs[6]).data[0].customdata[:, 0, 2], [0, 2])


def test_detail_click_pins_one_exact_measurement_without_rerunning_inference(tmp_path):
    make_checkpoint(tmp_path, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    class ExplodingModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError("detail clicks must not run the model")

    object.__setattr__(manager.loaded_state.checkpoint, "model", ExplodingModel())
    payload = json.dumps({
        "schema": "circuit-tracer.click.v1",
        "view": "detail",
        "token_position": 1,
        "dimension": 10,
    })

    pinned = app.select_clicked_token_callback(
        payload, "ffn_hidden", ["1"], False, "location", manager
    )

    assert manager.measurement_pin == app.MeasurementPin(1, 10)
    assert "**Pinned measurement:**" in pinned[1]
    assert "dimension `10`" in pinned[1]
    assert plot_value(pinned[5]).data[-1].customdata[0, 2] == 1
    assert plot_value(pinned[5]).data[-1].customdata[0, 4] == 10

    incompatible = app.select_location_callback(
        "output_norm", ["1"], False, "location", manager
    )
    assert manager.measurement_pin is None
    assert "**Pinned measurement cleared:**" in incompatible[1]
    assert "shape `3 \u00d7 8`" in incompatible[1]


def test_measurement_pin_persists_across_compatible_locations(tmp_path):
    make_checkpoint(tmp_path, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    detail_click = {
        "schema": "circuit-tracer.click.v1",
        "view": "detail",
        "token_position": 2,
        "dimension": 7,
    }
    app.select_clicked_token_callback(
        detail_click, "output_norm", ["2"], False, "location", manager
    )

    outputs = app.select_location_callback(
        "ffn_hidden", ["2"], False, "location", manager, "indexed"
    )

    assert manager.measurement_pin == app.MeasurementPin(2, 7)
    assert "dimension `7`" in outputs[1]
    assert "FFN hidden activation" in outputs[1]
    assert plot_value(outputs[5]).data[-1].customdata[0, 2] == 2
    assert plot_value(outputs[5]).data[-1].customdata[0, 4] == 7


def test_overview_click_cannot_create_a_measurement_pin(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    app.select_clicked_token_callback(
        {
            "schema": "circuit-tracer.click.v1",
            "view": "overview",
            "token_position": 1,
            "dimension": 4,
        },
        "output_norm", [], False, "location", manager,
    )

    assert manager.measurement_pin is None


def test_selection_before_analysis_reports_awaiting_state(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))

    token, explanation, stats, visible_range, magnitude, heatmap, distribution, diagram, *_ = app.select_location_callback(
        "output_norm", ["0"], False, "selected_point", manager
    )

    assert token["value"] is None
    assert explanation == app.INSPECT_AWAITING
    assert stats == ""
    assert visible_range == ""
    assert magnitude is None
    assert heatmap is None
    assert distribution is None
    assert "ct-selected" not in diagram


def test_invalid_token_positions_are_ignored(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    _, _, stats, _, magnitude, heatmap, distribution, *_ = app.select_location_callback(
        "output_norm", ["-1", "99"], False, "selected_point", manager
    )

    assert "Overview" in stats
    assert magnitude is None
    assert isinstance(plot_value(heatmap), go.Figure)
    assert distribution is None


def test_whole_model_normalization_uses_all_captured_locations(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    _, _, _, visible_range, _, heatmap, *_ = app.select_location_callback(
        "output_norm", ["0"], False, "whole_model", manager
    )
    lower, upper = app.all_heatmap_bounds(
        manager.inspection_session, clipped=False
    )

    assert f"`{lower:.4f}` to `{upper:.4f}`" in visible_range
    assert plot_value(heatmap).data[0].zmin == lower
    assert plot_value(heatmap).data[0].zmax == upper


def test_failed_analysis_preserves_the_stored_capture_and_view_outputs(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    previous_session = manager.inspection_session
    app.pin_comparison_callback("output_norm", ["0"], manager)
    previous_comparison = manager.comparison_state
    assert manager.pin_measurement("output_norm", 0, 1)
    previous_measurement = manager.measurement_pin

    outputs = app.analyze_and_inspect_callback("   ", manager)

    assert "Enter a prompt first" in outputs[0]
    assert outputs[1].startswith("AnalysisError:")
    assert all(value == app.gr.skip() for value in outputs[2:])
    assert manager.inspection_session is previous_session
    assert manager.comparison_state is previous_comparison
    assert manager.measurement_pin is previous_measurement


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


def test_failed_load_callback_preserves_active_session_and_ui_outputs(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    make_checkpoint(valid)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(valid))
    app.analyze_prompt_callback("hello , world", manager)
    app.pin_comparison_callback("output_norm", ["0"], manager)
    previous = manager.state_snapshot()

    outputs = app.load_and_reset_callback(str(tmp_path / "missing"), manager)

    assert len(outputs) == 29
    assert "previous model remains active" in outputs[0]
    assert outputs[1].startswith("CheckpointError:")
    assert all(value == app.gr.skip() for value in outputs[2:])
    current = manager.state_snapshot()
    assert current[0] is previous[0]
    assert current[1] is previous[1]
    assert current[2] is previous[2]


def test_successful_load_callback_resets_all_capture_dependent_outputs(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_checkpoint(first, seed=1)
    make_checkpoint(second, seed=2)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(first))
    app.analyze_prompt_callback("hello , world", manager)
    app.pin_comparison_callback("output_norm", ["0"], manager)
    assert manager.pin_measurement("output_norm", 0, 1)

    outputs = app.load_and_reset_callback(str(second), manager)

    assert len(outputs) == 29
    assert outputs[0] == "Model loaded successfully."
    assert outputs[1] == ""
    assert outputs[12]["value"] is None
    assert outputs[13]["value"] is None
    assert outputs[14]["value"] == "location"
    assert outputs[14]["interactive"]
    assert outputs[16] is False
    assert outputs[17] == app.INSPECT_AWAITING
    assert all(value is None for value in outputs[20:23])
    assert all(value is None for value in outputs[23:26])
    assert outputs[26] == "No comparison pinned."
    assert outputs[27]["capture"]["available"] is False
    assert outputs[28] == ""
    assert manager.inspection_session is None
    assert manager.comparison_state is None
    assert manager.measurement_pin is None


def test_analysis_candidate_cannot_commit_after_model_replacement(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    make_checkpoint(first, seed=1)
    make_checkpoint(second, seed=2)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(first))

    attempt = manager.prepare_analysis("hello")
    assert attempt.candidate is not None
    manager.load(str(second))

    assert not manager.commit_analysis(attempt.candidate)
    assert manager.loaded_state.checkpoint_path == second
    assert manager.inspection_session is None


def test_successful_analysis_replaces_capture_and_resets_comparison_and_views(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_and_inspect_callback("hello", manager)
    previous_session = manager.inspection_session
    app.pin_comparison_callback("output_norm", ["0"], manager)
    assert manager.pin_measurement("output_norm", 0, 1)

    outputs = app.analyze_and_inspect_callback("hello , world", manager)

    assert manager.inspection_session is not previous_session
    assert manager.comparison_state is None
    assert manager.measurement_pin is None
    assert outputs[6]["value"] == "output_norm"
    assert outputs[7]["value"] == []
    assert outputs[8]["value"] == "location"
    assert outputs[9]["value"] == "square"
    assert outputs[10] is False
    assert outputs[21] == "No comparison pinned."
    assert outputs[22]["comparison"]["pinned"] is False


def test_load_and_analysis_report_explicit_progress_stages(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    stages = []

    app.load_and_reset_callback(
        str(tmp_path),
        manager,
        lambda value, desc: stages.append(desc),
    )
    app.analyze_and_inspect_callback(
        "hello",
        manager,
        lambda value, desc: stages.append(desc),
    )

    assert stages == [
        "Validating checkpoint",
        "Loading model weights",
        "Rendering model details",
        "Checkpoint ready",
        "Validating prompt",
        "Tracing model internals",
        "Rendering capture",
        "Analysis ready",
    ]


def test_rendering_failure_does_not_commit_candidate_capture(tmp_path, monkeypatch):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_and_inspect_callback("hello", manager)
    previous_session = manager.inspection_session

    monkeypatch.setattr(
        app,
        "_successful_inspection_outputs",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("plot failed")),
    )
    outputs = app.analyze_and_inspect_callback("hello , world", manager)

    assert "previous capture remains active" in outputs[0]
    assert outputs[1] == "RuntimeError: plot failed"
    assert manager.inspection_session is previous_session


def test_pin_and_unpin_comparison_is_capture_local_and_diagnostics_are_stable(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    status, diagnostics = app.pin_comparison_callback("output_norm", ["0"], manager)
    assert "Pinned A" in status
    payload = json.loads(diagnostics)
    assert payload["schema"] == "circuit-tracer.diagnostics.v1"
    assert payload["comparison"] == {"location": "output_norm", "pinned": True, "shape": [3, 8]}
    assert diagnostics == app.diagnostics_json(manager, "inspect", status)

    app.clear_comparison_callback(manager)
    assert manager.comparison_state is None


def test_comparison_overview_renders_raw_a_b_and_exact_full_delta(tmp_path):
    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    locations = manager.inspection_session.analysis.capture.locations

    app.pin_comparison_callback("attention_update", [], manager)
    outputs = app.select_bridged_location_callback(
        "output_norm", [], False, "location", manager
    )

    figure_a = plot_value(outputs[9])
    figure_b = plot_value(outputs[10])
    delta = plot_value(outputs[11])
    np.testing.assert_array_equal(figure_a.data[0].z, locations["attention_update"])
    np.testing.assert_array_equal(figure_b.data[0].z, locations["output_norm"])
    np.testing.assert_allclose(
        delta.data[0].z,
        locations["output_norm"] - locations["attention_update"],
    )
    raw_bounds = app.shared_display_bounds(
        locations["attention_update"], locations["output_norm"]
    )
    assert (figure_a.data[0].zmin, figure_a.data[0].zmax) == raw_bounds
    assert (figure_b.data[0].zmin, figure_b.data[0].zmax) == raw_bounds
    assert (delta.data[0].zmin, delta.data[0].zmax) == app.display_bounds(
        locations["output_norm"] - locations["attention_update"]
    )
    assert "exact signed `B - A`" in outputs[12]
    assert 'ct-stage ct-pinned' in outputs[8]
    assert 'ct-stage ct-selected ct-comparison-b' in outputs[8]


def test_comparison_detail_shares_live_tokens_and_pooled_selection_bounds(tmp_path):
    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    locations = manager.inspection_session.analysis.capture.locations

    app.pin_comparison_callback("output_norm", [], manager)
    outputs = app.select_bridged_location_callback(
        "attention_update", ["2", "0"], False, "selection", manager, "indexed"
    )

    assert outputs[1]["value"] == ["0", "2"]
    figure_a = plot_value(outputs[9])
    figure_b = plot_value(outputs[10])
    delta = plot_value(outputs[11])
    np.testing.assert_array_equal(figure_a.data[0].z, locations["output_norm"][[0, 2]])
    np.testing.assert_array_equal(figure_b.data[0].z, locations["attention_update"][[0, 2]])
    pooled = app.shared_display_bounds(
        locations["output_norm"][[0, 2]],
        locations["attention_update"][[0, 2]],
    )
    assert (figure_a.data[0].zmin, figure_a.data[0].zmax) == pooled
    assert (figure_b.data[0].zmin, figure_b.data[0].zmax) == pooled
    np.testing.assert_allclose(
        delta.data[0].z,
        (locations["attention_update"] - locations["output_norm"])[[0, 2]],
    )
    assert manager.comparison_state.values.shape == (3, 8)


def test_unequal_ffn_width_keeps_raw_pair_and_disables_delta_precisely(tmp_path):
    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    app.pin_comparison_callback("ffn_hidden", ["1"], manager)
    outputs = app.select_bridged_location_callback(
        "output_norm", ["1"], False, "location", manager, "indexed"
    )

    assert plot_value(outputs[9]).data[0].z.shape == (1, 12)
    assert plot_value(outputs[10]).data[0].z.shape == (1, 8)
    assert plot_value(outputs[11]) is None
    assert outputs[11]["visible"] is False
    assert "A FFN hidden activation has shape `3 × 12`" in outputs[12]
    assert "B Final block output has shape `3 × 8`" in outputs[12]
    assert "no broadcasting, truncation, projection, or padding" in outputs[12]
    assert "Raw A and B remain available side by side" in outputs[12]


def test_selecting_a_again_or_unpinning_exits_comparison(tmp_path):
    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    app.pin_comparison_callback("output_norm", [], manager)
    app.select_bridged_location_callback(
        "attention_update", [], False, "location", manager
    )
    selected_a = app.select_bridged_location_callback(
        "output_norm", [], False, "location", manager
    )

    assert manager.comparison_state is None
    assert plot_value(selected_a[6]) is not None
    assert all(plot_value(selected_a[index]) is None for index in (9, 10, 11))
    assert selected_a[12] == "Comparison ended: A was selected again."

    app.pin_comparison_callback("output_norm", [], manager)
    app.select_bridged_location_callback(
        "attention_update", [], False, "location", manager
    )
    status, _ = app.clear_comparison_callback(manager)
    normal = app.select_location_callback(
        "attention_update", [], False, "location", manager
    )
    assert status == "Comparison A cleared."
    assert manager.comparison_state is None
    assert plot_value(normal[5]) is not None
    assert all(plot_value(normal[index]) is None for index in (8, 9, 10))


def test_empty_selection_is_overview_and_indexed_selection_is_detail(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    overview = app.select_location_callback("output_norm", [], False, "location", manager)
    assert overview[4] is None
    assert isinstance(plot_value(overview[5]), go.Figure)
    assert plot_value(overview[5]).data[0].z.shape == (3, 8)

    detail = app.select_location_callback(
        "output_norm", ["1"], False, "capture", manager, "indexed"
    )
    assert isinstance(detail[4], go.Figure)
    assert isinstance(plot_value(detail[5]), go.Figure)
    assert plot_value(detail[5]).data[0].z.shape == (1, 8)


def test_selection_scope_is_unavailable_only_while_selection_is_empty():
    empty = app.selection_scope_update([])
    selected = app.selection_scope_update(["0"])

    assert empty["choices"] == app.EMPTY_SELECTION_SCOPE_CHOICES
    assert empty["value"] == "location"
    assert empty["interactive"]
    assert selected["choices"] == app.SCALE_SCOPE_CHOICES
    assert selected["interactive"]


def test_clipping_warning_persists_and_overview_hover_values_stay_raw(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    source = manager.inspection_session.analysis.capture.locations["output_norm"].copy()

    _, _, _, visible_range, magnitude, heatmap, distribution, *_ = (
        app.select_location_callback(
            "output_norm", [], True, "location", manager
        )
    )

    assert "**Warning:**" in visible_range
    assert "Captured and hover values remain raw" in visible_range
    heatmap = plot_value(heatmap)
    assert "Display clipped" in heatmap.layout.title.subtitle.text
    np.testing.assert_array_equal(heatmap.data[0].z, source)
    assert magnitude is None
    assert distribution is None


def test_model_diagram_has_one_backbone_two_upper_branches_and_only_captures_clickable():
    diagram = app.render_model_diagram(
        tiny_config(),
        selected_key="output_norm",
        pinned_key="attention_update",
        comparison_key="output_norm",
        capture_available=True,
    )

    assert diagram.count('class="ct-backbone"') == 1
    assert "Uninterrupted residual backbone" in diagram
    assert "Attention branch" in diagram
    assert "Feed-forward branch" in diagram
    assert diagram.count('class="ct-branch-wire"') == 5
    assert diagram.count('<button type="button" class="ct-stage') == 10
    assert diagram.count('data-location-key="') == 10
    assert 'data-stage="vocabulary_projection"' in diagram
    assert 'data-location-key="vocabulary_projection"' not in diagram
    assert 'class="ct-stage ct-pinned" data-stage="attention_update"' in diagram
    assert 'class="ct-stage ct-selected ct-comparison-b" data-stage="output_norm"' in diagram
    assert diagram.index('data-stage="attention_residual"') < diagram.index(
        'data-stage="attention_norm"'
    )
    assert diagram.index('data-stage="ffn_residual"') < diagram.index(
        'data-stage="output_norm"'
    )
    assert 'ct-indicator-a">A</span>' in diagram
    assert 'ct-indicator-b">B</span>' in diagram


def test_grouped_location_navigator_uses_precise_labels_and_capture_shapes():
    navigator = app.render_location_navigator(
        tiny_config(feed_forward_dim=12),
        token_count=3,
        selected_key="output_norm",
        pinned_key="attention_update",
        comparison_key="output_norm",
    )

    for group in (
        "Embeddings",
        "Attention branch",
        "Feed-forward branch",
        "Residual backbone",
    ):
        assert f">{group}</h3>" in navigator
    for key in app.LOCATION_KEYS:
        assert navigator.count(f'data-location-key="{key}"') == 1
    assert "Token embeddings" in navigator
    assert "Attention residual sum" in navigator
    assert "First normalized output" in navigator
    assert "Final block output" in navigator
    assert '<span class="ct-location-shape">3 x 8</span>' in navigator
    assert '<span class="ct-location-shape">3 x 12</span>' in navigator
    assert 'class="ct-location-button ct-pinned" data-location-key="attention_update"' in navigator
    assert 'class="ct-location-button ct-selected ct-comparison-b" data-location-key="output_norm"' in navigator
    assert " disabled" not in navigator


def test_navigator_is_disabled_until_a_capture_exists():
    navigator = app.render_location_navigator(tiny_config(feed_forward_dim=12))

    assert navigator.count(" disabled") == len(app.LOCATION_KEYS)
    assert "T x 8" in navigator
    assert "T x 12" in navigator


def test_metadata_overview_keeps_full_tensor_statistics_available():
    values = np.asarray([[1.0, -1.0], [3.0, -3.0]])

    metadata = app.location_stats(
        app.location_spec("output_norm"), values, []
    )

    assert "**Shape:** `2 \u00d7 2`" in metadata
    assert "**Captured range:** `-3.0000` to `3.0000`" in metadata
    assert "**Captured mean / std:** `0.0000`" in metadata
    assert "**Selection count:** `0`" in metadata
    assert "Overview of every processed token" in metadata


def test_workbench_layout_contract_fixes_context_and_progressively_collapses_panels():
    css = app.APP_CSS

    assert "height: 100vh !important" in css
    assert ".ct-workbench" in css and "overflow: hidden !important" in css
    assert ".ct-canvas-scroll" in css and "overscroll-behavior: contain" in css
    assert "@media (max-width: 1280px) and (min-width: 761px)" in css
    assert "@media (max-width: 1080px) and (min-width: 761px)" in css
    assert "#ct-metadata-panel { display: none !important; }" in css
    assert "#ct-location-panel { display: none !important; }" in css
    assert "button[data-location-key]" in app.LOCATION_NAV_JS
    assert "trigger('click'" in app.LOCATION_NAV_JS
    assert "ct-force-open" in app.NAV_TOGGLE_JS
    assert "ct-force-open" in app.INSPECTOR_TOGGLE_JS
    assert ".ct-activation-plot" in css
    assert "overflow-y: auto !important" in css


def test_bridged_location_click_returns_native_selector_and_capture_views(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    outputs = app.select_bridged_location_callback(
        "ffn_hidden", [], False, "location", manager
    )

    assert len(outputs) == 13
    assert outputs[0]["value"] == "ffn_hidden"
    assert outputs[1]["value"] == []
    assert "FFN hidden activation" in outputs[2]
    assert isinstance(plot_value(outputs[6]), go.Figure)
    assert 'data-current-location="ffn_hidden"' in outputs[8]


def test_detail_mode_changes_keep_selection_literal_order_and_active_bounds(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)

    square = app.select_location_callback(
        "output_norm", ["2", "0"], False, "capture", manager, "square"
    )[5]
    indexed = app.select_location_callback(
        "output_norm", ["2", "0"], False, "capture", manager, "indexed"
    )[5]

    square = plot_value(square)
    indexed = plot_value(indexed)
    square_bounds = (square.data[0].zmin, square.data[0].zmax)
    indexed_bounds = (indexed.data[0].zmin, indexed.data[0].zmax)
    assert square_bounds == indexed_bounds
    np.testing.assert_array_equal(indexed.data[0].x, range(8))
    np.testing.assert_array_equal(indexed.data[0].customdata[:, 0, 2], [0, 2])


def test_browser_click_bridge_uses_structured_activation_identity_without_modifiers():
    script = app.CLICK_BRIDGE_JS

    assert "#ct-activation-plot" in script
    assert "circuit-tracer.activation.v1" in script
    assert "circuit-tracer.click.v1" in script
    assert "JSON.stringify(payload)" in script
    assert "view === 'overview'" in script
    assert "view === 'detail'" in script
    for modifier in ("ctrlKey", "shiftKey", "metaKey", "altKey"):
        assert modifier not in script


def test_diagnostics_popup_html_renders_read_only_view_with_local_plotly():
    html = app.diagnostics_popup_html()

    assert html.startswith("<!doctype html>")
    assert "Read-only" in html
    assert "Token magnitudes" in html
    assert "Selected-token distribution" in html
    assert app.PLOTLY_ASSET_PATH in html
    assert app.DIAGNOSTICS_SCHEMA in html
    assert "event.origin" in html
    assert "revision" in html
    assert "postMessage" in html
    assert "circuit-tracer.diagnostics-window.request" in html
    assert "localStorage" not in html
    assert "https://cdn.plot.ly" not in html


def test_diagnostics_popup_scripts_guard_origin_and_revision():
    assert app.DIAGNOSTICS_POPUP_NAME == "circuit-tracer-diagnostics"
    assert app.DIAGNOSTICS_POPUP_PATH == "/ct-diagnostics"
    assert app.DIAGNOSTICS_SCHEMA == "circuit-tracer.diagnostics-window.v1"

    popup_html = app.diagnostics_popup_html()
    assert "window.open" in app.OPEN_DIAGNOSTICS_JS
    assert app.DIAGNOSTICS_POPUP_NAME in app.OPEN_DIAGNOSTICS_JS
    assert "window.open" in app.OPEN_DIAGNOSTICS_JS
    assert app.DIAGNOSTICS_SCHEMA in popup_html
    assert "event.origin" in popup_html
    assert "revision" in popup_html


def test_diagnostics_routes_serve_popup_plotly_and_payload(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    manager = app.ModelManager(device_detector=lambda: fake_device())
    fastapi_app = FastAPI()
    app.register_diagnostics_http(fastapi_app, manager)

    client = TestClient(fastapi_app, base_url="http://127.0.0.1")
    page = client.get("/ct-diagnostics")
    assert page.status_code == 200
    assert app.DIAGNOSTICS_SCHEMA in page.text
    assert app.PLOTLY_ASSET_PATH in page.text

    plotly = client.get(app.PLOTLY_ASSET_PATH)
    assert plotly.status_code == 200
    assert plotly.headers["content-type"].startswith("application/javascript")
    assert b"Plotly" in plotly.content or b"function" in plotly.content

    status = client.get("/ct-diagnostics-status")
    assert status.status_code == 200
    assert "signature" in status.json()

    payload = client.get("/ct-diagnostics-payload")
    assert payload.status_code == 200
    parsed = json.loads(payload.text)
    assert parsed["schema"] == app.DIAGNOSTICS_SCHEMA
    assert isinstance(parsed["revision"], int)
    assert parsed["capture"]["available"] is False


def test_diagnostics_payload_is_atomic_versioned_and_clearable(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    payload = json.loads(manager.payload_text())
    assert payload["schema"] == app.DIAGNOSTICS_SCHEMA
    assert isinstance(payload["revision"], int)
    assert payload["capture"]["available"] is False
    assert payload["magnitudes"]["present"] is False
    assert payload["distribution"]["present"] is False
    initial_revision = payload["revision"]

    manager.load(str(tmp_path))
    app.analyze_and_inspect_callback("hello , world", manager)

    after = json.loads(manager.payload_text())
    assert after["revision"] > initial_revision
    assert after["capture"]["available"] is True
    assert after["capture"]["token_count"] == 3
    assert after["capture"]["location"]["key"] == "output_norm"
    assert after["magnitudes"]["present"] is True
    assert after["magnitudes"]["figure"] is not None
    assert after["distribution"]["present"] is False
    assert "Select one or more tokens" in after["distribution"]["note"]

    selected = json.loads(manager.payload_text("output_norm", ["0"]))
    assert selected["distribution"]["present"] is True
    assert selected["distribution"]["figure"] is not None
    assert selected["revision"] >= after["revision"]

    manager.clear_session()
    cleared = json.loads(manager.payload_text())
    assert cleared["capture"]["available"] is False
    assert cleared["magnitudes"]["present"] is False
    assert cleared["distribution"]["present"] is False
    assert cleared["revision"] > after["revision"]


def test_diagnostics_payload_view_is_recorded_from_selection(tmp_path):
    make_checkpoint(tmp_path)
    manager = app.ModelManager(device_detector=lambda: fake_device())
    manager.load(str(tmp_path))
    app.analyze_and_inspect_callback("hello , world", manager)

    selected = json.loads(manager.payload_text())
    assert selected["capture"]["available"] is True
    assert selected["capture"]["location"]["key"] == "output_norm"

    # navigate away then verify payload follows the recorded view
    app.select_bridged_location_callback(
        "ffn_hidden", ["1"], False, "location", manager
    )
    moved = json.loads(manager.payload_text())
    assert moved["capture"]["location"]["key"] == "ffn_hidden"
    assert moved["distribution"]["present"] is True
