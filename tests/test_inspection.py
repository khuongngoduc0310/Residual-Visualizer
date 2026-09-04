import numpy as np
import pytest
import tensorflow as tf

from checkpoint import load_checkpoint, save_checkpoint
from inspection import (
    CAPTURED_KEYS,
    DEFAULT_NODE_KEY,
    FAMILY_NODES,
    TRACE_ORDER,
    CapturedRun,
    InspectionError,
    capture_locations,
    family_keys,
    node_spec,
)
from model import ModelConfig, build_model


VOCABULARY = ["", "[UNK]", "hello", ",", "world", "!"]


def tiny_config():
    return ModelConfig(
        vocab_size=len(VOCABULARY),
        max_len=6,
        embedding_dim=8,
        num_heads=2,
        key_dim=4,
        feed_forward_dim=12,
        dropout_rate=0.5,
    )


@pytest.fixture(scope="module")
def loaded_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("checkpoint")
    tf.keras.utils.set_random_seed(9)
    config = tiny_config()
    model = build_model(config)
    save_checkpoint(directory, model, VOCABULARY, config)
    return load_checkpoint(directory)


def token_ids(prompt):
    vocabulary = {token: index for index, token in enumerate(VOCABULARY)}
    return tf.constant([vocabulary[token] for token in prompt.split()])


def test_capture_returns_every_tensor_with_expected_shape(loaded_checkpoint):
    ids = token_ids("hello , world !")
    captured = capture_locations(loaded_checkpoint, ids)

    assert isinstance(captured, CapturedRun)
    assert captured.token_count == 4
    assert set(captured.locations) == set(CAPTURED_KEYS)
    for key, tensor in captured.locations.items():
        if key == "attention_pattern":
            assert tensor.shape == (4, 4), key
        else:
            width = 12 if key == "ffn_hidden" else 8
            assert tensor.shape == (4, width), key
    assert captured.probabilities.shape == (4, len(VOCABULARY))


def test_single_token_capture_keeps_two_dimensional_shapes(loaded_checkpoint):
    captured = capture_locations(loaded_checkpoint, token_ids("hello"))

    assert captured.token_count == 1
    assert captured.probabilities.shape == (1, len(VOCABULARY))
    for key, tensor in captured.locations.items():
        if key == "attention_pattern":
            assert tensor.shape == (1, 1), key
        else:
            width = 12 if key == "ffn_hidden" else 8
            assert tensor.shape == (1, width), key


def test_captured_final_output_matches_normal_model_inference(
    loaded_checkpoint,
):
    ids = token_ids("hello , world !")
    expected = loaded_checkpoint.model(ids[None, :], training=False).numpy()[0]

    captured = capture_locations(loaded_checkpoint, ids)
    np.testing.assert_allclose(
        captured.probabilities,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )

    block_output = captured.locations["output_norm"]
    projection = loaded_checkpoint.model.get_layer("token_probabilities")
    projected = projection(tf.constant(block_output[None, :])).numpy()[0]
    np.testing.assert_allclose(projected, expected, rtol=1e-5, atol=1e-6)


def test_capture_is_deterministic_with_dropout_disabled(loaded_checkpoint):
    ids = token_ids("hello , world !")
    first = capture_locations(loaded_checkpoint, ids)
    second = capture_locations(loaded_checkpoint, ids)

    for key in CAPTURED_KEYS:
        np.testing.assert_array_equal(first.locations[key], second.locations[key])
    np.testing.assert_array_equal(first.probabilities, second.probabilities)


def test_capture_rejects_empty_token_ids(loaded_checkpoint):
    with pytest.raises(InspectionError, match="does not contain any tokens"):
        capture_locations(loaded_checkpoint, tf.constant([], dtype=tf.int32))


def test_capture_rejects_over_maximum_token_ids(loaded_checkpoint):
    with pytest.raises(InspectionError, match="at most 6"):
        capture_locations(
            loaded_checkpoint,
            tf.constant([2, 3, 4, 2, 3, 4, 2]),
        )


def test_node_catalog_is_complete_and_ordered():
    assert TRACE_ORDER == (
        "token_embeddings",
        "position_embeddings",
        "embedding",
        "attention_pattern",
        "attention_update",
        "attention_residual",
        "attention_norm",
        "ffn_hidden",
        "ffn_update",
        "ffn_residual",
        "output_norm",
        "readout",
    )
    assert DEFAULT_NODE_KEY == "output_norm"
    assert CAPTURED_KEYS == TRACE_ORDER[:-1]


def test_node_specs_have_families_kinds_and_explanations():
    from inspection import STREAM_NODES

    assert len(STREAM_NODES) == len(TRACE_ORDER)
    assert tuple(node.key for node in STREAM_NODES) == TRACE_ORDER
    for node in STREAM_NODES:
        assert node.family in set(FAMILY_NODES)
        assert node.explanation.strip()
        assert node.label.strip()
        assert node.key in family_keys(node.family)


def test_shared_scale_families_group_the_residual_path():
    assert family_keys("stream_raw") == (
        "embedding",
        "attention_residual",
        "ffn_residual",
    )
    assert family_keys("updates") == ("attention_update", "ffn_update")
    assert family_keys("stream_norm") == ("attention_norm", "output_norm")
    assert family_keys("components") == (
        "token_embeddings",
        "position_embeddings",
    )
    assert family_keys("hidden") == ("ffn_hidden",)
    assert family_keys("pattern") == ("attention_pattern",)
    assert family_keys("readout") == ("readout",)


def test_normalized_nodes_are_only_the_layer_norms():
    from inspection import STREAM_NODES

    normalized = {node.key for node in STREAM_NODES if node.normalized}
    assert normalized == {"attention_norm", "output_norm"}
    pattern = next(node for node in STREAM_NODES if node.key == "attention_pattern")
    assert pattern.feature_axis is False
    readout = next(node for node in STREAM_NODES if node.key == "readout")
    assert readout.feature_axis is False


def test_unknown_node_key_is_rejected():
    with pytest.raises(InspectionError, match="Unknown stream node"):
        node_spec("nonsense")
