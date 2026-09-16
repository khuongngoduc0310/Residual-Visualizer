import numpy as np
import pytest
import tensorflow as tf
from support import VOCABULARY, tiny_config, write_checkpoint

from checkpoint import load_checkpoint
from inspection import (
    ABLATABLE_NODES,
    BLOCK_NODE_STAGES,
    CAPTURED_KEYS,
    DEEMBEDDABLE_NODES,
    DEFAULT_NODE_KEY,
    FAMILY_NODES,
    RESIDUAL_STATES,
    SPINE_NODES,
    TRACE_ORDER,
    VOCAB_CONTRIBUTABLE_NODES,
    CapturedRun,
    InspectionError,
    block_node_key,
    capture_locations,
    family_keys,
    node_spec,
)
from model import NUM_TRANSFORMER_BLOCKS


@pytest.fixture(scope="module")
def loaded_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("checkpoint")
    write_checkpoint(
        directory,
        seed=9,
        config=tiny_config(feed_forward_dim=12, dropout_rate=0.5),
    )
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
        spec = node_spec(key)
        if spec.stage == "attention_pattern":
            assert tensor.shape == (4, 4), key
        else:
            width = 12 if spec.width_source == "ffn" else 8
            assert tensor.shape == (4, width), key
    assert captured.probabilities.shape == (4, len(VOCABULARY))


def test_single_token_capture_keeps_two_dimensional_shapes(loaded_checkpoint):
    captured = capture_locations(loaded_checkpoint, token_ids("hello"))

    assert captured.token_count == 1
    assert captured.probabilities.shape == (1, len(VOCABULARY))
    for key, tensor in captured.locations.items():
        spec = node_spec(key)
        if spec.stage == "attention_pattern":
            assert tensor.shape == (1, 1), key
        else:
            width = 12 if spec.width_source == "ffn" else 8
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


def test_captured_locations_preserve_the_pre_norm_residual_highway(
    loaded_checkpoint,
):
    captured = capture_locations(
        loaded_checkpoint, token_ids("hello , world !")
    ).locations
    residual_key = "embedding"
    for block_index in range(NUM_TRANSFORMER_BLOCKS):
        attention_residual = block_node_key(block_index, "attention_residual")
        attention_update = block_node_key(block_index, "attention_update")
        ffn_residual = block_node_key(block_index, "ffn_residual")
        ffn_update = block_node_key(block_index, "ffn_update")
        np.testing.assert_allclose(
            captured[attention_residual],
            captured[residual_key] + captured[attention_update],
            atol=1e-6,
        )
        np.testing.assert_allclose(
            captured[ffn_residual],
            captured[attention_residual] + captured[ffn_update],
            atol=1e-6,
        )
        block = loaded_checkpoint.model.get_layer(f"transformer_block_{block_index}")
        np.testing.assert_allclose(
            captured[block_node_key(block_index, "attention_input_norm")],
            block.attention_input_norm(captured[residual_key]),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            captured[block_node_key(block_index, "ffn_input_norm")],
            block.ffn_input_norm(captured[attention_residual]),
            atol=1e-6,
        )
        residual_key = ffn_residual
    final_norm = loaded_checkpoint.model.get_layer("final_output_layer_norm")
    np.testing.assert_allclose(
        captured["output_norm"],
        final_norm(captured[residual_key]),
        atol=1e-6,
    )


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
    block_keys = tuple(
        block_node_key(block_index, stage)
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for stage in BLOCK_NODE_STAGES
    )
    assert TRACE_ORDER == (
        "token_embeddings",
        "position_embeddings",
        "embedding",
        *block_keys,
        "output_norm",
        "readout",
    )
    assert DEFAULT_NODE_KEY == "output_norm"
    assert CAPTURED_KEYS == TRACE_ORDER[:-1]
    assert RESIDUAL_STATES == (
        "embedding",
        *(
            block_node_key(block_index, stage)
            for block_index in range(NUM_TRANSFORMER_BLOCKS)
            for stage in ("attention_residual", "ffn_residual")
        ),
    )
    assert SPINE_NODES == (*RESIDUAL_STATES, "output_norm")
    assert DEEMBEDDABLE_NODES == SPINE_NODES
    assert VOCAB_CONTRIBUTABLE_NODES == tuple(
        block_node_key(block_index, stage)
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for stage in ("attention_update", "ffn_update")
    )
    assert not set(VOCAB_CONTRIBUTABLE_NODES) & set(ABLATABLE_NODES)


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
    assert family_keys("stream_raw") == RESIDUAL_STATES
    assert family_keys("updates") == tuple(
        block_node_key(block_index, stage)
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for stage in ("attention_update", "ffn_update")
    )
    assert family_keys("norm") == (
        *(
            block_node_key(block_index, stage)
            for block_index in range(NUM_TRANSFORMER_BLOCKS)
            for stage in ("attention_input_norm", "ffn_input_norm")
        ),
        "output_norm",
    )
    assert family_keys("components") == (
        "token_embeddings",
        "position_embeddings",
    )
    assert family_keys("hidden") == tuple(
        block_node_key(block_index, "ffn_hidden")
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
    )
    assert family_keys("pattern") == tuple(
        block_node_key(block_index, "attention_pattern")
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
    )
    assert family_keys("readout") == ("readout",)


def test_normalized_nodes_are_only_the_layer_norms():
    from inspection import STREAM_NODES

    normalized = {node.key for node in STREAM_NODES if node.normalized}
    assert normalized == {
        *(
            block_node_key(block_index, stage)
            for block_index in range(NUM_TRANSFORMER_BLOCKS)
            for stage in ("attention_input_norm", "ffn_input_norm")
        ),
        "output_norm",
    }
    pattern = node_spec(block_node_key(0, "attention_pattern"))
    assert pattern.feature_axis is False
    assert pattern.block_index == 0
    assert pattern.stage == "attention_pattern"
    assert pattern.width_source == "none"
    readout = next(node for node in STREAM_NODES if node.key == "readout")
    assert readout.feature_axis is False


def test_unknown_node_key_is_rejected():
    with pytest.raises(InspectionError, match="Unknown stream node"):
        node_spec("nonsense")
