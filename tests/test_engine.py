"""Tests for the model lifecycle, loading, analysis, and ablation payloads."""

import json
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from support import (
    VOCABULARY,
    analyze_fixture,
    block0,
    loaded_manager,
    model_manager,
    tiny_config,
    write_checkpoint,
)

import engine
import inspection_views
from checkpoint import CONFIG_FILENAME, CheckpointError, LoadedCheckpoint
from model import ARCHITECTURE_NAME


def test_load_payload_loads_checkpoint_and_describes_the_model(tmp_path):
    config = write_checkpoint(tmp_path)
    manager = model_manager()

    payload = engine.load_model_payload(manager, str(tmp_path))

    assert payload["ok"]
    assert payload["loaded"]
    assert payload["status"] == "Model loaded successfully."
    assert payload["meta"]["path"] == str(tmp_path)
    assert payload["meta"]["architecture"] == ARCHITECTURE_NAME
    assert payload["meta"]["vocab_size"] == config.vocab_size
    assert payload["meta"]["max_len"] == config.max_len
    assert payload["meta"]["embedding_dim"] == config.embedding_dim
    assert payload["meta"]["num_heads"] == config.num_heads
    assert payload["meta"]["key_dim"] == config.key_dim
    assert payload["meta"]["feed_forward_dim"] == config.feed_forward_dim
    assert payload["meta"]["num_blocks"] == 3
    assert payload["meta"]["feed_forward_activity_l1"] == 1e-5
    assert payload["device_label"] == "CPU"
    assert "three_block_pre_norm_causal_lm" in payload["summary"]
    assert manager.loaded_state is not None
    assert manager.loaded_state.checkpoint.config == config


def test_load_payload_reports_missing_folder_and_has_no_model(tmp_path):
    manager = model_manager()

    payload = engine.load_model_payload(manager, str(tmp_path / "missing"))

    assert not payload["ok"]
    assert "Checkpoint could not be loaded" in payload["status"]
    assert "does not exist" in payload["status"]
    assert not payload["loaded"]
    assert payload["meta"]["path"] is None
    assert payload["device_label"] is None
    assert payload["summary"] is None
    assert manager.loaded_state is None


def test_load_payload_rejects_an_empty_folder_path():
    manager = model_manager()

    payload = engine.load_model_payload(manager, "   ")

    assert not payload["ok"]
    assert "Enter a checkpoint folder path" in payload["status"]
    assert manager.loaded_state is None


def test_failed_replacement_unloads_previous_model(tmp_path):
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    write_checkpoint(valid_path)
    manager = model_manager()

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
    write_checkpoint(first_path, seed=1)
    write_checkpoint(second_path, seed=2)
    clear_calls = []
    collect_calls = []
    manager = model_manager(
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
    write_checkpoint(tmp_path)
    (tmp_path / CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
    manager = model_manager()

    result = manager.load(str(tmp_path))

    assert not result.success
    assert "Checkpoint could not be loaded" in result.status
    assert "JSONDecodeError" not in result.status
    assert manager.loaded_state is None


def test_mismatched_checkpoint_is_reported_and_clears_previous_model(tmp_path):
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    write_checkpoint(valid_path)
    config_path = valid_path / CONFIG_FILENAME
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["model"]["feed_forward_dim"] = 12
    config_path.write_text(json.dumps(document), encoding="utf-8")
    manager = model_manager()

    result = manager.load(str(valid_path))

    assert not result.success
    assert "do not match" in result.status
    assert manager.loaded_state is None


def test_use_loaded_state_requires_and_yields_the_active_model(tmp_path):
    write_checkpoint(tmp_path)
    manager = model_manager()

    with pytest.raises(CheckpointError, match="Load a checkpoint"):
        with manager.use_loaded_state():
            pass

    manager.load(str(tmp_path))
    with manager.use_loaded_state() as state:
        assert state.checkpoint_path == tmp_path


def test_inspection_state_holds_the_model_lifecycle_lock():
    config = tiny_config()
    checkpoint = LoadedCheckpoint(model=object(), vocabulary=VOCABULARY, config=config)
    manager = model_manager(
        checkpoint_loader=lambda _directory: checkpoint,
        session_clearer=lambda: None,
        collector=lambda: 0,
    )
    assert manager.load("test-checkpoint").success

    entered = Event()
    release = Event()
    clear_done = Event()

    def hold_inspection_state():
        with manager.use_inspection_state() as (state, session):
            assert state is not None
            assert session is None
            entered.set()
            assert release.wait(timeout=2)

    def clear_manager():
        manager.clear()
        clear_done.set()

    inspection_thread = Thread(target=hold_inspection_state)
    inspection_thread.start()
    assert entered.wait(timeout=2)

    clear_thread = Thread(target=clear_manager)
    clear_thread.start()
    assert not clear_done.wait(timeout=0.05)

    release.set()
    inspection_thread.join(timeout=2)
    clear_thread.join(timeout=2)
    assert clear_done.is_set()
    assert manager.loaded_state is None


def test_cuda_device_requires_cuda_build_and_visible_gpu(monkeypatch):
    monkeypatch.setattr(
        engine.tf.sysconfig,
        "get_build_info",
        lambda: {"is_cuda_build": True},
    )
    monkeypatch.setattr(engine.tf.test, "is_built_with_cuda", lambda: True)
    monkeypatch.setattr(
        engine.tf.config,
        "list_physical_devices",
        lambda kind: [SimpleNamespace(name="GPU:0")] if kind == "GPU" else [],
    )
    monkeypatch.setattr(
        engine.tf.config.experimental,
        "get_device_details",
        lambda device: {"device_name": "Test NVIDIA GPU"},
    )

    device = engine.detect_compute_device()

    assert device.is_gpu
    assert device.tf_device == "/GPU:0"
    assert device.label == "CUDA GPU: Test NVIDIA GPU"


def test_cpu_device_does_not_claim_cuda(monkeypatch):
    monkeypatch.setattr(
        engine.tf.sysconfig,
        "get_build_info",
        lambda: {"is_cuda_build": False},
    )
    monkeypatch.setattr(engine.tf.test, "is_built_with_cuda", lambda: False)
    monkeypatch.setattr(engine.tf.config, "list_physical_devices", lambda kind: [])

    device = engine.detect_compute_device()

    assert not device.is_gpu
    assert device.tf_device == "/CPU:0"
    assert device.label == "CPU"


def test_analyze_without_model_reports_and_clears():
    manager = model_manager()

    payload = engine.analyze_prompt_payload(manager, "hello")

    assert not payload["ok"]
    assert "Load a checkpoint" in payload["status"]
    assert payload["token_count"] is None
    assert payload["tokens"] == []
    assert payload["next_tokens"] == []


def test_analyze_payload_renders_token_and_prediction_tables(tmp_path):
    manager = loaded_manager(tmp_path)

    payload = engine.analyze_prompt_payload(manager, "hello , world")

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
    write_checkpoint(tmp_path)
    manager = model_manager()
    manager.load(str(tmp_path))
    engine.analyze_prompt_payload(manager, "hello , world")
    assert manager.inspection_session is not None

    payload = engine.analyze_prompt_payload(manager, "   ")

    assert not payload["ok"]
    assert payload["status"] == "Enter a prompt first."
    assert payload["token_count"] is None
    assert payload["tokens"] == []
    assert payload["next_tokens"] == []
    assert manager.inspection_session is None


def test_ablation_requires_a_loaded_analysis():
    manager = model_manager()

    payload = engine.ablate_feature_payload(
        manager,
        block0("ffn_hidden"),
        [0],
        "zero",
        "all",
        None,
    )

    assert not payload["ok"]
    assert "Load a checkpoint" in payload["status"]


def test_ablation_stores_a_capture_and_exposes_comparisons(tmp_path):
    manager, _ = analyze_fixture(tmp_path)

    result = engine.ablate_feature_payload(
        manager,
        block0("ffn_hidden"),
        [0, 2],
        "zero",
        "token",
        1,
    )

    assert result["ok"]
    assert manager.inspection_session.ablated is not None
    assert result["ablation"]["dims"] == [0, 2]
    assert len(result["ablation"]["baseline_values"]) == 2

    diff = inspection_views.inspect_node_payload(
        manager, block0("ffn_hidden"), 1, "diff"
    )
    assert diff["state"] == "ready"
    assert diff["view"] == "diff"
    assert diff["map_figure"]["data"]
    assert diff["ablation"]["node_key"] == block0("ffn_hidden")

    readout = inspection_views.inspect_node_payload(manager, "readout", 1, "ablated")
    assert readout["readout_compare"] is not None
    assert readout["readout_compare_figure"]["data"]
    assert len(readout["position_effects"]) == 3
    session = manager.inspection_session
    for row in readout["readout_compare"]["movers"]:
        token_id = row["token_id"]
        baseline_probability = session.analysis.capture.probabilities[1, token_id]
        ablated_probability = session.ablated.analysis.capture.probabilities[
            1, token_id
        ]
        assert row["delta"] == pytest.approx(ablated_probability - baseline_probability)


def test_clearing_ablation_keeps_the_baseline_capture(tmp_path):
    manager, _ = analyze_fixture(tmp_path)
    engine.ablate_feature_payload(manager, "output_norm", [0], "zero", "all")
    assert manager.inspection_session.ablated is not None

    result = engine.clear_ablation_payload(manager)

    assert result == {"ok": True, "status": "Ablation cleared."}
    assert manager.inspection_session is not None
    assert manager.inspection_session.ablated is None
    baseline = inspection_views.inspect_node_payload(
        manager, "output_norm", 0, "baseline"
    )
    assert baseline["view"] == "baseline"


def test_loading_a_new_checkpoint_clears_the_stored_capture(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_checkpoint(first, seed=1)
    write_checkpoint(second, seed=2)
    manager = model_manager()
    manager.load(str(first))
    engine.analyze_prompt_payload(manager, "hello , world")
    assert manager.inspection_session is not None

    manager.load(str(second))

    assert manager.inspection_session is None
