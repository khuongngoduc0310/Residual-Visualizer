import numpy as np
import pytest
import tensorflow as tf

from preprocess import (
    build_text_vectorizer,
    pad_punctuation,
    prepare_training_batch,
    validate_vocabulary,
)


def test_pad_punctuation_is_deterministic():
    text = "Hello, don't [stop]\nnow."

    assert pad_punctuation(text) == "Hello , don ' t [ stop ] now ."


def test_vectorizer_preserves_saved_vocabulary_ids():
    vocabulary = ["", "[UNK]", "hello", ",", "world"]
    vectorizer = build_text_vectorizer(vocabulary=vocabulary)

    token_ids = vectorizer(tf.constant(["HELLO , missing world"])).numpy()

    np.testing.assert_array_equal(token_ids, [[2, 3, 1, 4]])


def test_training_batch_excludes_padding_targets_from_loss():
    vocabulary = ["", "[UNK]", "hello", "world"]
    vectorizer = build_text_vectorizer(
        vocabulary=vocabulary,
        output_sequence_length=5,
    )

    inputs, targets, weights = prepare_training_batch(
        tf.constant(["hello world"]), vectorizer
    )

    np.testing.assert_array_equal(inputs.numpy(), [[2, 3, 0, 0]])
    np.testing.assert_array_equal(targets.numpy(), [[3, 0, 0, 0]])
    np.testing.assert_array_equal(weights.numpy(), [[1.0, 0.0, 0.0, 0.0]])


def test_vocabulary_rejects_duplicate_tokens():
    with pytest.raises(ValueError, match="unique"):
        validate_vocabulary(["", "[UNK]", "hello", "hello"])
