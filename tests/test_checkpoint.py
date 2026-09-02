import json

import numpy as np
import pytest
import tensorflow as tf

from checkpoint import (
    CONFIG_FILENAME,
    CHECKPOINT_FORMAT_VERSION,
    SUPPORTED_KERAS_VERSION,
    SUPPORTED_TENSORFLOW_VERSION,
    VOCABULARY_FILENAME,
    WEIGHTS_FILENAME,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from model import ModelConfig, build_model
from preprocess import build_text_vectorizer, pad_punctuation


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


def save_tiny_checkpoint(path):
    with tf.device("/CPU:0"):
        tf.keras.utils.set_random_seed(9)
        config = tiny_config()
        language_model = build_model(config)
        sample = tf.constant([[2, 3, 4]])
        expected = language_model(sample, training=False).numpy()
        save_checkpoint(path, language_model, VOCABULARY, config)
    return expected, sample


def test_checkpoint_round_trip_preserves_predictions_and_token_ids(tmp_path):
    expected, sample = save_tiny_checkpoint(tmp_path)

    with tf.device("/CPU:0"):
        loaded = load_checkpoint(tmp_path)
        actual = loaded.model(sample, training=False).numpy()

    assert {item.name for item in tmp_path.iterdir()} == {
        WEIGHTS_FILENAME,
        VOCABULARY_FILENAME,
        CONFIG_FILENAME,
    }
    assert loaded.vocabulary == VOCABULARY
    np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-7)

    document = json.loads(
        (tmp_path / CONFIG_FILENAME).read_text(encoding="utf-8")
    )
    assert document["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert document["tensorflow_version"] == SUPPORTED_TENSORFLOW_VERSION
    assert document["keras_version"] == SUPPORTED_KERAS_VERSION

    original_vectorizer = build_text_vectorizer(vocabulary=VOCABULARY)
    loaded_vectorizer = build_text_vectorizer(vocabulary=loaded.vocabulary)
    text = pad_punctuation("HELLO, missing world!")
    np.testing.assert_array_equal(
        original_vectorizer(tf.constant([text])).numpy(),
        loaded_vectorizer(tf.constant([text])).numpy(),
    )


def test_load_refuses_missing_checkpoint_file(tmp_path):
    with pytest.raises(CheckpointError, match="Missing checkpoint file"):
        load_checkpoint(tmp_path)


def test_load_refuses_an_older_checkpoint_format(tmp_path):
    save_tiny_checkpoint(tmp_path)
    config_path = tmp_path / CONFIG_FILENAME
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["format_version"] = 1
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="Unsupported checkpoint format"):
        load_checkpoint(tmp_path)


def test_load_refuses_a_checkpoint_from_another_keras_version(tmp_path):
    save_tiny_checkpoint(tmp_path)
    config_path = tmp_path / CONFIG_FILENAME
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["keras_version"] = "3.12.0"
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="Checkpoint Keras version"):
        load_checkpoint(tmp_path)


def test_load_refuses_vocabulary_size_mismatch(tmp_path):
    save_tiny_checkpoint(tmp_path)
    (tmp_path / VOCABULARY_FILENAME).write_text(
        json.dumps(VOCABULARY[:-1]), encoding="utf-8"
    )

    with pytest.raises(CheckpointError, match="vocabulary size"):
        load_checkpoint(tmp_path)


def test_load_refuses_wrong_special_tokens(tmp_path):
    save_tiny_checkpoint(tmp_path)
    vocabulary = ["<PAD>", "[UNK]", *VOCABULARY[2:]]
    (tmp_path / VOCABULARY_FILENAME).write_text(
        json.dumps(vocabulary), encoding="utf-8"
    )

    with pytest.raises(CheckpointError, match="token 0"):
        load_checkpoint(tmp_path)


def test_load_explains_incompatible_weight_shapes(tmp_path):
    save_tiny_checkpoint(tmp_path)
    config_path = tmp_path / CONFIG_FILENAME
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["model"]["feed_forward_dim"] = 12
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="files do not match"):
        load_checkpoint(tmp_path)


def test_load_refuses_behavior_settings_mixed_with_other_weights(tmp_path):
    save_tiny_checkpoint(tmp_path)
    config_path = tmp_path / CONFIG_FILENAME
    document = json.loads(config_path.read_text(encoding="utf-8"))
    document["model"]["layer_norm_epsilon"] = 1.0
    config_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(CheckpointError, match="files do not match"):
        load_checkpoint(tmp_path)


def test_save_refuses_model_settings_that_do_not_match_model(tmp_path):
    model_config = tiny_config(layer_norm_epsilon=1e-6)
    language_model = build_model(model_config)
    incorrect_config = tiny_config(layer_norm_epsilon=1.0)

    with pytest.raises(CheckpointError, match="does not match"):
        save_checkpoint(
            tmp_path,
            language_model,
            VOCABULARY,
            incorrect_config,
        )


def test_checkpoint_refuses_unexpected_files(tmp_path):
    (tmp_path / "extra.txt").write_text("unexpected", encoding="utf-8")
    language_model = build_model(tiny_config())

    with pytest.raises(CheckpointError, match="Unexpected checkpoint file"):
        save_checkpoint(tmp_path, language_model, VOCABULARY, tiny_config())


def test_checkpoint_refuses_to_overwrite_existing_export(tmp_path):
    save_tiny_checkpoint(tmp_path)
    language_model = build_model(tiny_config())

    with pytest.raises(CheckpointError, match="must be empty"):
        save_checkpoint(tmp_path, language_model, VOCABULARY, tiny_config())


def test_load_refuses_unexpected_files(tmp_path):
    save_tiny_checkpoint(tmp_path)
    (tmp_path / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(CheckpointError, match="Unexpected checkpoint file"):
        load_checkpoint(tmp_path)
