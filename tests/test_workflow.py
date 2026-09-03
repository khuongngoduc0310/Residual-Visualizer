import numpy as np
import plotly.graph_objects as go
import tensorflow as tf

import app
from checkpoint import load_checkpoint, save_checkpoint
from charts import token_magnitudes
from inspection import LOCATION_KEYS
from model import ModelConfig, build_model


VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]


def test_cpu_checkpoint_to_all_internal_charts(tmp_path, monkeypatch):
    """Exercise the complete local path using an explicitly CPU-bound model."""
    monkeypatch.setattr(app, "detect_compute_device", lambda: app.ComputeDevice(
        label="CPU", tf_device="/CPU:0", is_gpu=False
    ))
    config = ModelConfig(
        vocab_size=len(VOCABULARY),
        max_len=6,
        embedding_dim=8,
        num_heads=2,
        key_dim=4,
        feed_forward_dim=12,
        dropout_rate=0.5,
    )
    with tf.device("/CPU:0"):
        tf.keras.utils.set_random_seed(9)
        model = build_model(config)
        save_checkpoint(tmp_path, model, VOCABULARY, config)
        checkpoint = load_checkpoint(tmp_path)

    manager = app.ModelManager(
        device_detector=lambda: app.ComputeDevice(
            label="CPU", tf_device="/CPU:0", is_gpu=False
        )
    )
    loaded = manager.load(str(tmp_path))
    assert loaded.success
    assert not manager.loaded_state.device.is_gpu

    outputs = app.analyze_and_inspect_callback("Hello, world!", manager)
    analysis = manager.inspection_session.analysis
    assert analysis.token_count == 4
    assert len(analysis.next_tokens) == 5
    assert set(analysis.capture.locations) == set(LOCATION_KEYS)

    token_labels = [f"{token.position}: {token.text}" for token in analysis.tokens]
    for location_key in LOCATION_KEYS:
        values = analysis.capture.locations[location_key]
        assert values.shape[0] == analysis.token_count
        magnitude = app.render_token_magnitudes(values, token_labels, 3)
        heatmap = app.render_activation_heatmap(values, token_labels, 3, False)
        distribution = app.render_token_distribution(values[3], 3, token_labels[3])
        np.testing.assert_allclose(magnitude.data[0].y, token_magnitudes(values))
        np.testing.assert_allclose(heatmap.data[0].z, values)
        np.testing.assert_allclose(
            distribution.data[0].x,
            values[3],
        )
        assert all(isinstance(figure, go.Figure) for figure in (magnitude, heatmap, distribution))

    assert outputs[5]["value"] == "output_norm"
    assert outputs[6]["value"] == "3"
    assert checkpoint.config == config
