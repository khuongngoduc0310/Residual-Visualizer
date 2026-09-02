import numpy as np
import pytest
import tensorflow as tf

import model as model_module
from model import ModelConfig, build_model


def tiny_config(**changes):
    values = {
        "vocab_size": 7,
        "max_len": 6,
        "embedding_dim": 8,
        "num_heads": 2,
        "key_dim": 4,
        "feed_forward_dim": 8,
        "dropout_rate": 0.0,
    }
    values.update(changes)
    return ModelConfig(**values)


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


def test_config_rejects_non_finite_layer_norm_epsilon():
    with pytest.raises(ValueError, match="layer_norm_epsilon"):
        tiny_config(layer_norm_epsilon=float("nan"))


def test_model_has_one_prediction_output():
    language_model = build_model(tiny_config())

    assert language_model.output_shape == (None, None, 7)


def test_trailing_right_padding_does_not_change_real_token_outputs():
    tf.keras.utils.set_random_seed(4)
    language_model = build_model(tiny_config())

    short = language_model(tf.constant([[2, 3, 4]]), training=False).numpy()
    padded = language_model(
        tf.constant([[2, 3, 4, 0, 0]]), training=False
    ).numpy()

    np.testing.assert_allclose(short, padded[:, :3, :], rtol=1e-5, atol=1e-6)
