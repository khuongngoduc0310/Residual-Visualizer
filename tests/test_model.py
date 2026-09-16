import numpy as np
import pytest
import tensorflow as tf
from support import tiny_config as _tiny_config

import model as model_module
from model import (
    ARCHITECTURE_NAME,
    NUM_TRANSFORMER_BLOCKS,
    ModelConfig,
    build_model,
    causal_attention_mask,
)


def tiny_config(**changes):
    """Model-layer config with the wider vocabulary these tests assert on."""
    return _tiny_config(vocab_size=7, **changes)


def test_model_import_has_no_prebuilt_model():
    assert not hasattr(model_module, "gpt")


def test_config_requires_attention_width_to_match_model_width():
    with pytest.raises(ValueError, match="num_heads.*key_dim.*embedding_dim"):
        tiny_config(key_dim=3)


def test_default_attention_width_matches_default_model_width():
    config = ModelConfig(vocab_size=10)

    assert config.num_heads == 2
    assert config.key_dim == 128
    assert config.num_heads * config.key_dim == config.embedding_dim
    assert config.num_blocks == 3
    assert config.feed_forward_activity_l1 == 1e-5


def test_config_rejects_non_finite_layer_norm_epsilon():
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        tiny_config(layer_norm_epsilon=float("nan"))


def test_config_requires_three_blocks_and_valid_activity_l1():
    with pytest.raises(ValueError, match="num_blocks must be 3"):
        tiny_config(num_blocks=2)
    with pytest.raises(ValueError, match="feed_forward_activity_l1"):
        tiny_config(feed_forward_activity_l1=-1.0)
    with pytest.raises(ValueError, match="feed_forward_activity_l1"):
        tiny_config(feed_forward_activity_l1=0.0)
    with pytest.raises(ValueError, match="num_blocks must be a positive integer"):
        tiny_config(num_blocks=3.0)


def test_model_has_one_prediction_output():
    language_model = build_model(tiny_config())

    assert language_model.name == "three_block_pre_norm_causal_lm"
    assert ARCHITECTURE_NAME == "three_block_pre_norm_causal_lm"
    assert NUM_TRANSFORMER_BLOCKS == 3
    assert [
        layer.name
        for layer in language_model.layers
        if layer.name.startswith("transformer_block_")
    ] == ["transformer_block_0", "transformer_block_1", "transformer_block_2"]
    assert language_model.output_shape == (None, None, 7)


def test_block_implements_pre_norm_equations_with_final_output_norm():
    tf.keras.utils.set_random_seed(7)
    language_model = build_model(tiny_config())
    token_ids = tf.constant([[2, 3, 4]])
    embedding_layer = language_model.get_layer("token_and_position_embedding")
    embeddings = embedding_layer(token_ids)
    residual = embeddings
    for block_index in range(NUM_TRANSFORMER_BLOCKS):
        block = language_model.get_layer(f"transformer_block_{block_index}")
        steps, attention_scores = block.call_steps(residual, training=False)
        expected_attention_input = block.attention_input_norm(residual)
        np.testing.assert_allclose(
            steps["attention_input_norm"], expected_attention_input, atol=1e-6
        )

        mask = causal_attention_mask(1, 3, 3, tf.bool)
        expected_attention_update, expected_scores = block.attn(
            expected_attention_input,
            expected_attention_input,
            attention_mask=mask,
            return_attention_scores=True,
            training=False,
        )
        np.testing.assert_allclose(
            steps["attention_update"], expected_attention_update, atol=1e-6
        )
        np.testing.assert_allclose(attention_scores, expected_scores, atol=1e-6)
        np.testing.assert_allclose(
            steps["attention_residual"],
            residual + steps["attention_update"],
            atol=1e-6,
        )

        expected_ffn_input = block.ffn_input_norm(steps["attention_residual"])
        np.testing.assert_allclose(
            steps["ffn_input_norm"], expected_ffn_input, atol=1e-6
        )
        expected_hidden = block.ffn_1(expected_ffn_input)
        expected_ffn_update = block.ffn_2(expected_hidden)
        np.testing.assert_allclose(steps["ffn_hidden"], expected_hidden, atol=1e-6)
        np.testing.assert_allclose(steps["ffn_update"], expected_ffn_update, atol=1e-6)
        np.testing.assert_allclose(
            steps["ffn_residual"],
            steps["attention_residual"] + steps["ffn_update"],
            atol=1e-6,
        )
        residual = steps["ffn_residual"]

    final_norm = language_model.get_layer("final_output_layer_norm")
    output_norm = final_norm(residual)
    projection = language_model.get_layer("token_probabilities")
    expected_probabilities = projection(output_norm)
    actual_probabilities = language_model(token_ids, training=False)
    np.testing.assert_allclose(actual_probabilities, expected_probabilities, atol=1e-6)


def test_trailing_right_padding_does_not_change_real_token_outputs():
    tf.keras.utils.set_random_seed(4)
    language_model = build_model(tiny_config())

    short = language_model(tf.constant([[2, 3, 4]]), training=False).numpy()
    padded = language_model(tf.constant([[2, 3, 4, 0, 0]]), training=False).numpy()

    np.testing.assert_allclose(short, padded[:, :3, :], rtol=1e-5, atol=1e-6)


def test_activity_l1_excludes_right_padding_and_matches_hidden_values():
    tf.keras.utils.set_random_seed(5)
    config = tiny_config(feed_forward_activity_l1=1e-5)
    language_model = build_model(config)
    token_ids = tf.constant([[2, 3, 0, 0], [4, 5, 0, 0]])

    language_model(token_ids, training=False)
    actual_losses = np.asarray([float(loss) for loss in language_model.losses])

    residual = language_model.get_layer("token_and_position_embedding")(token_ids)
    expected_losses = []
    for block_index in range(NUM_TRANSFORMER_BLOCKS):
        block = language_model.get_layer(f"transformer_block_{block_index}")
        steps, _ = block.call_steps(residual, training=False)
        expected_losses.append(
            config.feed_forward_activity_l1
            * float(tf.reduce_sum(tf.abs(steps["ffn_hidden"][:, :2, :])))
            / 2.0
        )
        residual = steps["ffn_residual"]

    assert actual_losses.shape == (NUM_TRANSFORMER_BLOCKS,)
    np.testing.assert_allclose(actual_losses, expected_losses, atol=1e-7)

    language_model(tf.constant([[2, 3], [4, 5]]), training=False)
    unpadded_losses = np.asarray([float(loss) for loss in language_model.losses])
    np.testing.assert_allclose(actual_losses, unpadded_losses, atol=1e-7)
