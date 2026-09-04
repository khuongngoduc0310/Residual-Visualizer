import numpy as np
import tensorflow as tf

import app
from checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from charts import grid_shape
from inspection import STREAM_NODES, TRACE_ORDER, capture_locations
from model import ModelConfig, build_model


VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]


def test_cpu_checkpoint_to_all_stream_node_views(tmp_path):
    """Exercise the complete local path using an explicitly CPU-bound model."""
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

    analyzed = app.analyze_prompt_payload(manager, "Hello, world!")
    assert analyzed["ok"]
    analysis = manager.inspection_session.analysis
    assert analysis.token_count == 4
    assert len(analysis.next_tokens) == 5
    assert set(analysis.capture.locations) == set(TRACE_ORDER[:-1])

    token_labels = [f"{token.position}: {token.text}" for token in analysis.tokens]
    for node in STREAM_NODES:
        if node.kind == "readout":
            continue
        values = analysis.capture.locations[node.key]
        assert values.shape[0] == analysis.token_count
        payload = app.inspect_node_payload(manager, node.key, 3, False)
        assert payload["state"] == "ready"
        assert payload["selected_position"] == 3
        if node.kind == "pattern":
            assert payload["pattern_figure"]["data"]
            np.testing.assert_allclose(
                payload["pattern_figure"]["data"][0]["z"], values
            )
            continue
        map_figure = payload["map_figure"]
        tile = payload["tile"]
        assert tile is not None
        assert (tile["rows"], tile["cols"]) == grid_shape(values.shape[1])
        assert len(map_figure["data"]) == 1
        assert "data" in map_figure
        assert "layout" in map_figure
        token_count = analysis.token_count
        rows, cols = tile["rows"], tile["cols"]
        z = np.asarray(map_figure["data"][0]["z"], dtype=float)
        assert z.shape == (rows, cols * token_count)
        assert not np.isnan(z).any()
        for index in range(token_count):
            start = index * cols
            np.testing.assert_allclose(
                z[:, start : start + cols],
                values[index].reshape(rows, cols),
            )

    default_inspect = app.inspect_node_payload(manager)
    assert default_inspect["selected_position"] == 3
    assert default_inspect["node"]["key"] == "output_norm"
    assert checkpoint.config == config


def _captured(seed=4):
    config = ModelConfig(
        vocab_size=len(VOCABULARY),
        max_len=6,
        embedding_dim=8,
        num_heads=2,
        key_dim=4,
        feed_forward_dim=12,
        dropout_rate=0.5,
    )
    tf.keras.utils.set_random_seed(seed)
    model = build_model(config)
    checkpoint = LoadedCheckpoint(model=model, vocabulary=VOCABULARY, config=config)
    return capture_locations(checkpoint, tf.constant([2, 3, 4]))


def test_embedding_components_sum_to_the_stream_input():
    captured = _captured()

    np.testing.assert_allclose(
        captured.locations["token_embeddings"]
        + captured.locations["position_embeddings"],
        captured.locations["embedding"],
    )


def test_residual_updates_equal_the_stream_difference():
    captured = _captured()

    locations = captured.locations
    np.testing.assert_allclose(
        locations["attention_update"],
        locations["attention_residual"] - locations["embedding"],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        locations["ffn_update"],
        locations["ffn_residual"] - locations["attention_norm"],
        atol=1e-6,
    )


def test_attention_pattern_rows_are_normalized_and_causal():
    captured = _captured()

    pattern = captured.locations["attention_pattern"]
    assert pattern.shape == (3, 3)
    np.testing.assert_allclose(pattern.sum(axis=1), np.ones(3), atol=1e-5)
    assert np.all(pattern[np.triu_indices(3, k=1)] == 0.0)
