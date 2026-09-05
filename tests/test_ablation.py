import numpy as np
import pytest
import tensorflow as tf

from checkpoint import load_checkpoint, save_checkpoint
from inspection import (
    AblationError,
    AblationSpec,
    capture_locations,
)
from model import ModelConfig, build_model


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


@pytest.fixture(scope="module")
def loaded_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ablation-checkpoint")
    tf.keras.utils.set_random_seed(23)
    config = tiny_config()
    model = build_model(config)
    save_checkpoint(directory, model, VOCABULARY, config)
    return load_checkpoint(directory)


def token_ids():
    return tf.constant([2, 3, 4], dtype=tf.int32)


def test_zero_ablation_replaces_exactly_one_output_dimension(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="output_norm",
        dim=2,
        mode="zero",
        scope="token",
        position=1,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    np.testing.assert_array_equal(
        ablated.locations["output_norm"][1, :2],
        baseline.locations["output_norm"][1, :2],
    )
    assert ablated.locations["output_norm"][1, 2] == 0.0
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][1, 3:],
        baseline.locations["output_norm"][1, 3:],
    )
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][[0, 2]],
        baseline.locations["output_norm"][[0, 2]],
    )


def test_ffn_hidden_ablation_recomputes_the_output_write(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="ffn_hidden",
        dim=1,
        mode="zero",
        scope="token",
        position=2,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    block = loaded_checkpoint.model.get_layer("transformer_block")
    hidden_value = baseline.locations["ffn_hidden"][2, spec.dim]
    output_column = block.ffn_2.kernel.numpy()[spec.dim, :]
    expected_update_delta = -hidden_value * output_column
    actual_update_delta = (
        ablated.locations["ffn_update"][2]
        - baseline.locations["ffn_update"][2]
    )
    np.testing.assert_allclose(
        actual_update_delta,
        expected_update_delta,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        ablated.locations["ffn_update"][:2],
        baseline.locations["ffn_update"][:2],
    )


def test_mean_ablation_uses_leave_one_out_value(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="output_norm",
        dim=4,
        mode="mean",
        scope="token",
        position=1,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)
    expected = np.mean(np.delete(baseline.locations["output_norm"][:, 4], 1))

    assert ablated.locations["output_norm"][1, 4] == expected
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][[0, 2]],
        baseline.locations["output_norm"][[0, 2]],
    )


def test_all_token_mean_ablation_makes_the_column_constant(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="ffn_hidden",
        dim=3,
        mode="mean",
        scope="all",
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)
    expected = np.mean(baseline.locations["ffn_hidden"][:, 3])

    np.testing.assert_array_equal(
        ablated.locations["ffn_hidden"][:, 3],
        np.full(3, expected),
    )


def test_embedding_ablation_respects_causal_prefix(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="embedding",
        dim=0,
        mode="zero",
        scope="token",
        position=1,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    for key in ("token_embeddings", "position_embeddings"):
        np.testing.assert_array_equal(
            ablated.locations[key],
            baseline.locations[key],
        )
    for key, value in baseline.locations.items():
        if key in {"attention_pattern"}:
            np.testing.assert_array_equal(ablated.locations[key][:1], value[:1])
        elif key != "embedding":
            np.testing.assert_array_equal(ablated.locations[key][:1], value[:1])


def test_ablation_rejects_invalid_dimension_and_single_token_mean(loaded_checkpoint):
    with pytest.raises(AblationError, match="outside the width"):
        capture_locations(
            loaded_checkpoint,
            token_ids(),
            AblationSpec(
                node_key="ffn_hidden",
                dim=loaded_checkpoint.config.feed_forward_dim,
                mode="zero",
                scope="all",
            ),
        )

    with pytest.raises(AblationError, match="another token"):
        capture_locations(
            loaded_checkpoint,
            tf.constant([2], dtype=tf.int32),
            AblationSpec(
                node_key="output_norm",
                dim=0,
                mode="mean",
                scope="token",
                position=0,
            ),
        )
