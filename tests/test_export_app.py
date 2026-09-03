"""App-level export integration and browser/figure contract tests.

These tests import the real ``app`` module (so they exercise the same HTTP
routes, options payload, dialog strings, and figure-export scripts that the
workbench ships) and use a tiny CPU checkpoint exactly like ``test_app.py``.
The browser-level behaviour is asserted on the embedded scripts and the
diagnostics popup HTML, matching the project's existing testing approach.
"""

import json

import numpy as np
import pytest
import tensorflow as tf

import app
import export as ex
from checkpoint import save_checkpoint
from model import ModelConfig, build_model


VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]


def tiny_config(**changes):
    values = {
        "vocab_size": len(VOCABULARY),
        "max_len": 6,
        "embedding_dim": 8,
        "num_heads": 2,
        "key_dim": 4,
        "feed_forward_dim": 12,
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


def fake_device():
    return app.ComputeDevice(label="CPU", tf_device="/CPU:0", is_gpu=False)


def loaded_manager(path):
    manager = app.ModelManager(device_detector=fake_device)
    manager.load(str(path))
    return manager


def analyzed_manager(path, prompt="hello , world"):
    manager = loaded_manager(path)
    app.analyze_prompt_callback(prompt, manager)
    return manager


def test_export_options_are_disabled_until_a_capture_exists():
    manager = app.ModelManager(device_detector=fake_device)
    payload = app.export_options_payload(manager)

    assert payload["schema"] == ex.EXPORT_OPTIONS_SCHEMA
    assert payload["available"] is False
    assert payload["capture"]["available"] is False
    assert all(scope["available"] is False for scope in payload["scopes"])
    assert all(figure["available"] is False for figure in payload["figures"])
    assert payload["delta"]["available"] is False
    assert "Load a checkpoint" in payload["message"]


def test_export_options_gate_selection_scope_on_tokens_and_comparison_on_pin(tmp_path):
    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = app.ModelManager(device_detector=fake_device)
    manager.load(str(tmp_path))
    app.analyze_prompt_callback("hello , world", manager)
    app.select_bridged_location_callback("output_norm", [], False, "location", manager)

    before = app.export_options_payload(manager)
    by_scope = {scope["value"]: scope for scope in before["scopes"]}
    by_figure = {figure["value"]: figure for figure in before["figures"]}
    assert by_scope[ex.SCOPE_SELECTION]["available"] is False
    assert "Select one or more" in by_scope[ex.SCOPE_SELECTION]["reason"]
    assert by_scope[ex.SCOPE_LOCATION]["available"] is True
    assert by_scope[ex.SCOPE_COMPARISON]["available"] is False
    assert "Pin a location" in by_scope[ex.SCOPE_COMPARISON]["reason"]
    assert by_figure[ex.FIGURE_ACTIVATION]["available"] is True
    assert by_figure[ex.FIGURE_MAGNITUDES]["available"] is False
    assert by_figure[ex.FIGURE_DISTRIBUTION]["available"] is False
    assert before["delta"]["available"] is False

    app.select_bridged_location_callback("output_norm", ["1"], False, "location", manager)
    after = app.export_options_payload(manager)
    by_scope = {scope["value"]: scope for scope in after["scopes"]}
    by_figure = {figure["value"]: figure for figure in after["figures"]}
    assert by_scope[ex.SCOPE_SELECTION]["available"] is True
    assert by_figure[ex.FIGURE_MAGNITUDES]["available"] is True
    assert by_figure[ex.FIGURE_DISTRIBUTION]["available"] is True
    assert after["selection"]["count"] == 1


def test_incompatible_and_compatible_comparison_option_states(tmp_path):
    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = analyzed_manager(tmp_path)
    app.pin_comparison_callback("attention_update", [], manager)
    app.select_bridged_location_callback("ffn_hidden", [], False, "location", manager)

    payload = app.export_options_payload(manager)
    by_scope = {scope["value"]: scope for scope in payload["scopes"]}
    assert by_scope[ex.SCOPE_COMPARISON]["available"] is True
    assert payload["comparison"]["active"] is True
    assert payload["comparison"]["shapes_equal"] is False
    assert payload["delta"]["available"] is False
    assert "3 x 12" in payload["delta"]["reason"]

    app.select_bridged_location_callback("output_norm", [], False, "location", manager)
    compatible = app.export_options_payload(manager)
    assert compatible["comparison"]["active"] is True
    assert compatible["comparison"]["shapes_equal"] is True
    assert compatible["delta"]["available"] is True
    assert compatible["comparison"]["a_key"] == "attention_update"
    assert compatible["comparison"]["b_key"] == "output_norm"


def test_export_http_routes_build_csv_round_trip_after_explicit_request(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = analyzed_manager(tmp_path)
    app.select_bridged_location_callback("output_norm", ["1", "0"], False, "location", manager)
    fastapi_app = FastAPI()
    app.register_diagnostics_http(fastapi_app, manager)
    client = TestClient(fastapi_app, base_url="http://127.0.0.1")

    options = client.get("/ct-export-options")
    assert options.status_code == 200
    parsed = options.json()
    assert parsed["available"] is True
    assert parsed["selection"]["count"] == 2

    payload = client.post(
        "/ct-export",
        json={"scope": ex.SCOPE_SELECTION, "format": ex.FORMAT_CSV},
    )
    assert payload.status_code == 200
    assert payload.headers["content-type"].startswith("text/csv")
    assert "circuit-tracer_export_selection" in payload.headers["content-disposition"]
    metadata_text, table = ex.parse_csv_content(payload.content)
    header, rows = table[0], table[1:]
    assert header == list(ex.CSV_COLUMNS_SINGLE)
    assert len(rows) == 2 * 8
    metadata = json.loads(metadata_text)
    assert metadata["scope"] == ex.SCOPE_SELECTION
    assert metadata["format"] == ex.FORMAT_CSV
    source = manager.inspection_session.analysis.capture.locations["output_norm"]
    for row in rows:
        assert float(row[5]) == source[int(row[0]), int(row[4])]


def test_export_http_refuses_invalid_scope_and_missing_selection(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    make_checkpoint(tmp_path)
    manager = analyzed_manager(tmp_path)
    fastapi_app = FastAPI()
    app.register_diagnostics_http(fastapi_app, manager)
    client = TestClient(fastapi_app, base_url="http://127.0.0.1")

    rejected = client.post(
        "/ct-export",
        json={"scope": ex.SCOPE_SELECTION, "format": ex.FORMAT_CSV},
    )
    assert rejected.status_code == 400
    assert "Select one or more" in rejected.json()["reason"]

    unknown = client.post(
        "/ct-export",
        json={"scope": "capture", "format": ex.FORMAT_CSV},
    )
    assert unknown.status_code == 400
    assert "Unknown export scope" in unknown.json()["reason"]

    image = client.post(
        "/ct-export",
        json={"scope": ex.SCOPE_LOCATION, "format": ex.FORMAT_PNG},
    )
    assert image.status_code == 400
    assert "PNG/SVG images are downloaded in the browser" in image.json()["reason"]


def test_export_http_rejects_non_object_large_and_nonlocal_requests(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    make_checkpoint(tmp_path)
    manager = analyzed_manager(tmp_path)
    fastapi_app = FastAPI()
    app.register_diagnostics_http(fastapi_app, manager)
    local = TestClient(fastapi_app, base_url="http://127.0.0.1")

    non_object = local.post("/ct-export", json=[])
    assert non_object.status_code == 400
    assert "JSON object" in non_object.json()["reason"]

    oversized = local.post(
        "/ct-export",
        content=b"x" * (64 * 1024 + 1),
        headers={"content-type": "application/json"},
    )
    assert oversized.status_code == 413

    foreign = TestClient(fastapi_app, base_url="http://example.test")
    assert foreign.get("/ct-export-options").status_code == 403
    assert foreign.get("/ct-diagnostics-payload").status_code == 403


def test_export_http_comparison_delta_refusal_and_compatible_delta_npz(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    make_checkpoint(tmp_path, embedding_dim=8, feed_forward_dim=12)
    manager = analyzed_manager(tmp_path)
    app.pin_comparison_callback("attention_update", [], manager)
    app.select_bridged_location_callback("ffn_hidden", [], False, "location", manager)
    fastapi_app = FastAPI()
    app.register_diagnostics_http(fastapi_app, manager)
    client = TestClient(fastapi_app, base_url="http://127.0.0.1")

    refused = client.post(
        "/ct-export",
        json={
            "scope": ex.SCOPE_COMPARISON,
            "format": ex.FORMAT_NPZ,
            "include_delta": True,
        },
    )
    assert refused.status_code == 400
    assert "Delta export refused" in refused.json()["reason"]

    pair = client.post(
        "/ct-export",
        json={
            "scope": ex.SCOPE_COMPARISON,
            "format": ex.FORMAT_NPZ,
            "include_delta": False,
        },
    )
    assert pair.status_code == 200
    arrays, metadata_text = ex.load_npz_bytes(pair.content)
    assert "delta" not in arrays
    assert json.loads(metadata_text)["delta"]["available"] is False

    app.select_bridged_location_callback("output_norm", [], False, "location", manager)
    compatible = client.post(
        "/ct-export",
        json={
            "scope": ex.SCOPE_COMPARISON,
            "format": ex.FORMAT_NPZ,
            "include_delta": True,
        },
    )
    assert compatible.status_code == 200
    locations = manager.inspection_session.analysis.capture.locations
    a_expected = np.asarray(locations["attention_update"], dtype=float)
    b_expected = np.asarray(locations["output_norm"], dtype=float)
    arrays, metadata_text = ex.load_npz_bytes(compatible.content)
    np.testing.assert_array_equal(arrays["values_a"], a_expected)
    np.testing.assert_array_equal(arrays["values_b"], b_expected)
    np.testing.assert_array_equal(arrays["delta"], b_expected - a_expected)
    assert json.loads(metadata_text)["delta"]["available"] is True


def test_main_workbench_figure_export_script_uses_local_plotly_only():
    script = app.EXPORT_IMAGE_JS

    assert "Plotly.downloadImage" in script
    assert "'png'" in script and "'svg'" in script
    assert "/ct-assets/plotly.min.js" in script
    for host in (
        "ct-activation-plot",
        "ct-comparison-a-plot",
        "ct-comparison-b-plot",
        "ct-delta-plot",
        "ct-magnitude-plot",
        "ct-distribution-plot",
    ):
        assert host in script
    assert "kaleido" not in script.lower()
    assert "cdn.plot.ly" not in script
    assert "cdnjs" not in script


def test_export_dialog_html_is_an_accessible_modal_with_explicit_choices():
    html = app.EXPORT_DIALOG_HTML

    assert "<dialog" in html and "id=\"ct-export-dialog\"" in html
    assert "aria-labelledby" in html
    assert "role=\"status\"" in html and "aria-live" in html
    for scope in ex.EXPORT_SCOPES:
        assert f'value="{scope}"' in html
    for format_ in ex.EXPORT_FORMATS:
        assert f'value="{format_}"' in html
    for target in ex.FIGURE_TARGETS:
        assert f'value="{target}"' in html
    assert 'name="ct-export-scope"' in html
    assert 'name="ct-export-format"' in html
    assert 'name="ct-export-figure"' in html
    assert 'id="ct-export-delta"' in html
    assert 'name="ct-export-delta"' in html
    assert "explicit" in html or "explicitly" in html


def test_export_dialog_js_refreshes_options_and_disables_invalid_choices():
    script = app.OPEN_EXPORT_DIALOG_JS

    assert script.lstrip().startswith("() =>")
    assert not script.rstrip().endswith(")();")
    assert "/ct-export-options" in script
    assert "/ct-export" in script
    assert "showModal" in script
    assert "disabled = !item.available" in script
    assert "input.disabled" in script
    assert "ctDownloadExportImage" in script or "downloadImage" in script
    assert "include_delta" in script


def test_diagnostics_window_ships_browser_image_export_with_local_plotly():
    html = app.diagnostics_popup_html()

    assert "Plotly.downloadImage" in html
    assert 'data-export-graph="mag"' in html
    assert 'data-export-graph="dist"' in html
    assert 'data-export-format="png"' in html
    assert 'data-export-format="svg"' in html
    assert "fig-export" in html
    assert app.PLOTLY_ASSET_PATH in html
    assert "Read-only" in html
    assert "https://cdn.plot.ly" not in html
    assert "kaleido" not in html.lower()
    assert "localStorage" not in html


def test_create_app_accepts_export_wiring_without_launching():
    demo = app.create_app(app.ModelManager(device_detector=fake_device))
    configured_scripts = [
        dependency["js"]
        for dependency in demo.get_config_file()["dependencies"]
        if dependency.get("js")
    ]

    assert isinstance(demo, app.gr.Blocks)
    assert "ct-export-dialog" in app.EXPORT_DIALOG_HTML
    assert configured_scripts
    assert all(script.lstrip().startswith("() =>") for script in configured_scripts)
