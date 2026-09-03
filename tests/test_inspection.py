import numpy as np
import pytest
import tensorflow as tf

from checkpoint import load_checkpoint, save_checkpoint
from inspection import (
    ATTENTION,
    DEFAULT_LOCATION_KEY,
    EMBEDDING,
    FFN,
    LOCATION_KEYS,
    RESIDUAL_STREAM,
    CapturedRun,
    InspectionError,
    capture_locations,
    location_choices,
    location_spec,
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


def test_capture_returns_every_location_with_expected_shape(loaded_checkpoint):
    ids = token_ids("hello , world !")
    captured = capture_locations(loaded_checkpoint, ids)

    assert isinstance(captured, CapturedRun)
    assert captured.token_count == 4
    assert set(captured.locations) == set(LOCATION_KEYS)
    for key, tensor in captured.locations.items():
        width = 12 if key == "ffn_hidden" else 8
        assert tensor.shape == (4, width), key
    assert captured.probabilities.shape == (4, len(VOCABULARY))


def test_single_token_capture_keeps_two_dimensional_shapes(loaded_checkpoint):
    captured = capture_locations(loaded_checkpoint, token_ids("hello"))

    assert captured.token_count == 1
    assert captured.probabilities.shape == (1, len(VOCABULARY))
    for key, tensor in captured.locations.items():
        width = 12 if key == "ffn_hidden" else 8
        assert tensor.shape == (1, width), key


def test_embedding_components_sum_to_pre_attention_residual(loaded_checkpoint):
    captured = capture_locations(loaded_checkpoint, token_ids("hello , world !"))

    np.testing.assert_allclose(
        captured.locations["token_embedding"]
        + captured.locations["position_embedding"],
        captured.locations["embedding"],
    )


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

    for key in LOCATION_KEYS:
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


def test_location_catalog_is_complete_and_classified():
    assert LOCATION_KEYS == (
        "token_embedding",
        "position_embedding",
        "embedding",
        "attention_update",
        "attention_residual",
        "attention_norm",
        "ffn_hidden",
        "ffn_update",
        "ffn_residual",
        "output_norm",
    )
    assert DEFAULT_LOCATION_KEY == "output_norm"


def test_location_specs_have_categories_and_explanations():
    from inspection import LOCATIONS

    assert len(LOCATIONS) == 10
    assert tuple(spec.key for spec in LOCATIONS) == LOCATION_KEYS
    for spec in LOCATIONS:
        assert spec.category in {RESIDUAL_STREAM, EMBEDDING, ATTENTION, FFN}
        assert spec.explanation.strip()
        assert spec.label.strip()
    assert {spec.key: spec.category for spec in LOCATIONS} == {
        "token_embedding": EMBEDDING,
        "position_embedding": EMBEDDING,
        "embedding": RESIDUAL_STREAM,
        "attention_update": ATTENTION,
        "attention_residual": RESIDUAL_STREAM,
        "attention_norm": RESIDUAL_STREAM,
        "ffn_hidden": FFN,
        "ffn_update": FFN,
        "ffn_residual": RESIDUAL_STREAM,
        "output_norm": RESIDUAL_STREAM,
    }
    assert {spec.key: spec.normalized for spec in LOCATIONS} == {
        "token_embedding": False,
        "position_embedding": False,
        "embedding": False,
        "attention_update": False,
        "attention_residual": False,
        "attention_norm": True,
        "ffn_hidden": False,
        "ffn_update": False,
        "ffn_residual": False,
        "output_norm": True,
    }


def test_location_choices_are_labeled_and_ordered():
    choices = location_choices()

    assert len(choices) == 10
    assert [value for _, value in choices] == list(LOCATION_KEYS)
    assert all(" \u00b7 " in label for label, _ in choices)
    assert choices[-1] == ("Residual Stream \u00b7 Final block output", "output_norm")


def test_unknown_location_key_is_rejected():
    with pytest.raises(InspectionError, match="Unknown internal location"):
        location_spec("nonsense")
