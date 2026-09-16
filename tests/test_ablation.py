import numpy as np
import pytest
import tensorflow as tf
from support import block0, write_checkpoint

from checkpoint import load_checkpoint
from inspection import (
    AblationError,
    AblationSpec,
    block_node_key,
    capture_locations,
    node_spec,
)


@pytest.fixture(scope="module")
def loaded_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ablation-checkpoint")
    write_checkpoint(directory, seed=23)
    return load_checkpoint(directory)


def token_ids():
    return tf.constant([2, 3, 4], dtype=tf.int32)


def test_zero_ablation_replaces_selected_output_dimensions(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="output_norm",
        dims=(2, 4),
        mode="zero",
        scope="token",
        position=1,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    np.testing.assert_array_equal(
        ablated.locations["output_norm"][1, :2],
        baseline.locations["output_norm"][1, :2],
    )
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][1, [2, 4]],
        np.zeros(2),
    )
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][1, [0, 1, 3, 5, 6, 7]],
        baseline.locations["output_norm"][1, [0, 1, 3, 5, 6, 7]],
    )
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][[0, 2]],
        baseline.locations["output_norm"][[0, 2]],
    )


def test_ffn_hidden_ablation_recomputes_the_output_write(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key=block0("ffn_hidden"),
        dims=(1, 5),
        mode="zero",
        scope="token",
        position=2,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    block = loaded_checkpoint.model.get_layer("transformer_block_0")
    hidden_values = baseline.locations[block0("ffn_hidden")][2, list(spec.dims)]
    output_columns = block.ffn_2.kernel.numpy()[list(spec.dims), :]
    expected_update_delta = -hidden_values @ output_columns
    actual_update_delta = (
        ablated.locations[block0("ffn_update")][2]
        - baseline.locations[block0("ffn_update")][2]
    )
    np.testing.assert_allclose(
        actual_update_delta,
        expected_update_delta,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        ablated.locations[block0("ffn_update")][:2],
        baseline.locations[block0("ffn_update")][:2],
    )


@pytest.mark.parametrize(
    ("node_key", "unchanged_key", "changed_key"),
    [
        ("attention_input_norm", None, "attention_update"),
        ("ffn_input_norm", "attention_residual", "ffn_update"),
    ],
)
def test_pre_norm_branch_input_ablation_propagates_only_downstream(
    loaded_checkpoint,
    node_key,
    unchanged_key,
    changed_key,
):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key=block0(node_key),
        dims=tuple(range(loaded_checkpoint.config.embedding_dim)),
        mode="zero",
        scope="all",
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    np.testing.assert_array_equal(
        ablated.locations[
            "embedding" if unchanged_key is None else block0(unchanged_key)
        ],
        baseline.locations[
            "embedding" if unchanged_key is None else block0(unchanged_key)
        ],
    )
    np.testing.assert_array_equal(
        ablated.locations[block0(node_key)],
        np.zeros_like(ablated.locations[block0(node_key)]),
    )
    assert not np.allclose(
        ablated.locations[block0(changed_key)],
        baseline.locations[block0(changed_key)],
    )
    np.testing.assert_allclose(
        ablated.locations[block0("attention_residual")],
        ablated.locations["embedding"] + ablated.locations[block0("attention_update")],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        ablated.locations[block0("ffn_residual")],
        ablated.locations[block0("attention_residual")]
        + ablated.locations[block0("ffn_update")],
        atol=1e-6,
    )


def test_last_block_ablation_leaves_earlier_blocks_unchanged(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    target = block_node_key(2, "ffn_input_norm")
    spec = AblationSpec(
        node_key=target,
        dims=tuple(range(loaded_checkpoint.config.embedding_dim)),
        mode="zero",
        scope="all",
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    for key, values in baseline.locations.items():
        node = node_spec(key)
        if key in {"token_embeddings", "position_embeddings", "embedding"} or (
            node.block_index is not None and node.block_index < 2
        ):
            np.testing.assert_array_equal(ablated.locations[key], values)
    np.testing.assert_array_equal(
        ablated.locations[target], np.zeros_like(ablated.locations[target])
    )


def test_first_block_ablation_propagates_through_later_blocks(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key=block0("ffn_hidden"),
        dims=tuple(range(loaded_checkpoint.config.feed_forward_dim)),
        mode="zero",
        scope="all",
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)

    assert not np.allclose(
        ablated.locations[block_node_key(2, "ffn_residual")],
        baseline.locations[block_node_key(2, "ffn_residual")],
    )
    assert not np.allclose(ablated.probabilities, baseline.probabilities)


def test_mean_ablation_uses_leave_one_out_value(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="output_norm",
        dims=(4, 6),
        mode="mean",
        scope="token",
        position=1,
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)
    expected = np.mean(
        np.delete(baseline.locations["output_norm"][:, list(spec.dims)], 1, axis=0),
        axis=0,
    )

    np.testing.assert_allclose(
        ablated.locations["output_norm"][1, list(spec.dims)],
        expected,
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        ablated.locations["output_norm"][[0, 2]],
        baseline.locations["output_norm"][[0, 2]],
    )


def test_all_token_mean_ablation_makes_the_column_constant(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key=block0("ffn_hidden"),
        dims=(3, 5),
        mode="mean",
        scope="all",
    )
    ablated = capture_locations(loaded_checkpoint, token_ids(), spec)
    expected = np.mean(
        baseline.locations[block0("ffn_hidden")][:, list(spec.dims)],
        axis=0,
    )

    np.testing.assert_array_equal(
        ablated.locations[block0("ffn_hidden")][:, list(spec.dims)],
        np.broadcast_to(expected, (3, len(spec.dims))),
    )


def test_embedding_ablation_respects_causal_prefix(loaded_checkpoint):
    baseline = capture_locations(loaded_checkpoint, token_ids())
    spec = AblationSpec(
        node_key="embedding",
        dims=(0, 2),
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
                node_key=block0("ffn_hidden"),
                dims=(loaded_checkpoint.config.feed_forward_dim,),
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
                dims=(0,),
                mode="mean",
                scope="token",
                position=0,
            ),
        )


def test_ablation_deduplicates_dimensions_and_rejects_empty_lists():
    spec = AblationSpec(
        node_key="output_norm",
        dims=(4, 2, 4),
        mode="zero",
        scope="all",
    )

    assert spec.dims == (2, 4)
    with pytest.raises(AblationError, match="At least one"):
        AblationSpec(
            node_key="output_norm",
            dims=(),
            mode="zero",
            scope="all",
        )
