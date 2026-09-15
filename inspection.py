from dataclasses import dataclass, replace
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from checkpoint import LoadedCheckpoint
from model import NUM_TRANSFORMER_BLOCKS


class InspectionError(ValueError):
    pass


DEFAULT_NODE_KEY = "output_norm"

# Kinds: component, stream, update, ln, hidden, pattern, readout.
# Families share a display color scale across their nodes:
#   components {token, position embeddings}
#   stream_raw {input, after-attention, after-FFN}
#   updates    {attention update, FFN update}
#   norm       {attention input, FFN input, final output norms}
#   hidden     {FFN hidden}   (single, non-negative sequential scale)
#   pattern    {attention pattern} (fixed 0..1)
#   readout    {probabilities / entropy}
COMPONENTS = "components"
STREAM_RAW = "stream_raw"
UPDATES = "updates"
NORM = "norm"
HIDDEN = "hidden"
PATTERN = "pattern"
READOUT = "readout"


@dataclass(frozen=True)
class StreamNode:
    key: str
    label: str
    kind: str
    family: str
    explanation: str
    normalized: bool = False
    feature_axis: bool = True
    block_index: int | None = None
    stage: str | None = None
    width_source: str = "model"


SINGLE_BLOCK_NODE_TEMPLATES: Tuple[StreamNode, ...] = (
    StreamNode(
        key="token_embeddings",
        label="Token embeddings",
        kind="component",
        family=COMPONENTS,
        explanation=(
            "The embedding lookup for every prompt token, before any position "
            "information is added. Summed with the position embeddings it "
            "forms the residual stream input."
        ),
    ),
    StreamNode(
        key="position_embeddings",
        label="Position embeddings",
        kind="component",
        family=COMPONENTS,
        explanation=(
            "The learned embedding for each token position, identical for "
            "every prompt at that index. Summed with the token embeddings it "
            "forms the residual stream input."
        ),
    ),
    StreamNode(
        key="embedding",
        label="Residual stream \u00b7 input",
        kind="stream",
        family=STREAM_RAW,
        explanation=(
            "Token plus position embeddings: the residual stream as it enters "
            "the first transformer block. Its attention branch reads a "
            "normalized copy of this value."
        ),
    ),
    StreamNode(
        key="attention_input_norm",
        label="Layer norm - attention input",
        kind="ln",
        family=NORM,
        normalized=True,
        explanation=(
            "The normalized copy of the residual stream read by causal "
            "attention. The raw incoming stream bypasses this branch and is "
            "preserved for the residual addition."
        ),
    ),
    StreamNode(
        key="attention_pattern",
        label="Causal attention pattern",
        kind="pattern",
        family=PATTERN,
        explanation=(
            "Attention weights computed from the normalized attention input "
            "and averaged over heads. Each query row sums to about one; "
            "future key columns are masked by causality."
        ),
        feature_axis=False,
        width_source="none",
    ),
    StreamNode(
        key="attention_update",
        label="Attention output \u2192 residual",
        kind="update",
        family=UPDATES,
        explanation=(
            "The value causal attention computes from its normalized input. "
            "Adding it to the raw incoming stream produces this block's "
            "post-attention residual."
        ),
    ),
    StreamNode(
        key="attention_residual",
        label="Residual stream \u00b7 after attention",
        kind="stream",
        family=STREAM_RAW,
        explanation=(
            "The residual stream after the attention update is added back to "
            "the block input. Feature magnitudes here combine the incoming "
            "stream with what attention just wrote."
        ),
    ),
    StreamNode(
        key="ffn_input_norm",
        label="Layer norm - FFN input",
        kind="ln",
        family=NORM,
        normalized=True,
        explanation=(
            "The normalized copy of the post-attention residual read by the "
            "feed-forward network. The raw residual bypasses this branch and "
            "is preserved for the next residual addition."
        ),
    ),
    StreamNode(
        key="ffn_hidden",
        label="FFN hidden (ReLU)",
        kind="hidden",
        family=HIDDEN,
        explanation=(
            "The raw ReLU hidden activation of the feed-forward network. "
            "Values are non-negative; zero marks a dead neuron for that "
            "token."
        ),
        width_source="ffn",
    ),
    StreamNode(
        key="ffn_update",
        label="FFN output \u2192 residual",
        kind="update",
        family=UPDATES,
        explanation=(
            "The value the feed-forward network writes into the residual "
            "stream. Adding this to the raw post-attention residual produces "
            "the residual stream after the FFN."
        ),
    ),
    StreamNode(
        key="ffn_residual",
        label="Residual stream \u00b7 after FFN",
        kind="stream",
        family=STREAM_RAW,
        explanation=(
            "The raw post-attention residual plus the FFN update. This is the "
            "transformer block output; it flows into the next block or, after "
            "the final block, the readout normalization."
        ),
    ),
    StreamNode(
        key="output_norm",
        label="Layer norm - readout input",
        kind="ln",
        family=NORM,
        normalized=True,
        explanation=(
            "A final model-level normalization of the raw block output. The "
            "vocabulary projection reads this value to predict the next token."
        ),
    ),
    StreamNode(
        key="readout",
        label="Readout \u00b7 next-token probabilities",
        kind="readout",
        family=READOUT,
        explanation=(
            "For the selected token position, the softmax distribution over "
            "the vocabulary induced by the final block output. The entropy "
            "strip shows how confident the model is at every token."
        ),
        feature_axis=False,
        width_source="none",
    ),
)

BLOCK_NODE_STAGES: Tuple[str, ...] = (
    "attention_input_norm",
    "attention_pattern",
    "attention_update",
    "attention_residual",
    "ffn_input_norm",
    "ffn_hidden",
    "ffn_update",
    "ffn_residual",
)


def block_node_key(block_index: int, stage: str) -> str:
    return f"blocks.{block_index}.{stage}"


_GLOBAL_PREFIX_NODES = SINGLE_BLOCK_NODE_TEMPLATES[:3]
_BLOCK_NODE_TEMPLATES = SINGLE_BLOCK_NODE_TEMPLATES[3:-2]
_GLOBAL_SUFFIX_NODES = SINGLE_BLOCK_NODE_TEMPLATES[-2:]
STREAM_NODES: Tuple[StreamNode, ...] = (
    *_GLOBAL_PREFIX_NODES,
    *(
        replace(
            node,
            key=block_node_key(block_index, node.key),
            label=f"Block {block_index + 1} - {node.label}",
            block_index=block_index,
            stage=node.key,
        )
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for node in _BLOCK_NODE_TEMPLATES
    ),
    *_GLOBAL_SUFFIX_NODES,
)

NODE_BY_KEY: Dict[str, StreamNode] = {node.key: node for node in STREAM_NODES}
TRACE_ORDER: Tuple[str, ...] = tuple(node.key for node in STREAM_NODES)
# Every tensor actually captured in the single analysis run. The readout node
# is virtual: it is derived from the stored probabilities matrix.
CAPTURED_KEYS: Tuple[str, ...] = tuple(
    key for key in TRACE_ORDER if key != "readout"
)

# Raw states carried by the residual highway.
RESIDUAL_STATES: Tuple[str, ...] = (
    "embedding",
    *(
        block_node_key(block_index, stage)
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for stage in ("attention_residual", "ffn_residual")
    ),
)

# Main diagram path, including the final readout preparation node.
SPINE_NODES: Tuple[str, ...] = (
    *RESIDUAL_STATES,
    "output_norm",
)

DEEMBEDDABLE_NODES: Tuple[str, ...] = (*RESIDUAL_STATES, "output_norm")

VOCAB_CONTRIBUTABLE_NODES: Tuple[str, ...] = tuple(
    block_node_key(block_index, stage)
    for block_index in range(NUM_TRANSFORMER_BLOCKS)
    for stage in ("attention_update", "ffn_update")
)

ABLATABLE_NODES: Tuple[str, ...] = (
    "embedding",
    *(
        block_node_key(block_index, stage)
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for stage in (
            "attention_input_norm",
            "attention_residual",
            "ffn_input_norm",
            "ffn_hidden",
            "ffn_residual",
        )
    ),
    "output_norm",
)
ABLATION_MODES: Tuple[str, ...] = ("zero", "mean")
ABLATION_SCOPES: Tuple[str, ...] = ("token", "all")

@dataclass(frozen=True)
class BranchSpec:
    key: str
    label: str
    reads: str
    adds_before: str
    path: Tuple[str, ...]
    observables: Tuple[str, ...] = ()
    kind: str = "attention"
    block_index: int = 0
    side: str = "above"


# Links between consecutive SPINE_NODES plus the readout at the end.
#   embedding -> attention_residual  : attention add junction
#   attention_residual -> ffn_residual : FFN add junction
#   ffn_residual -> output_norm      : layer norm (in line)
#   output_norm -> readout           : projection + softmax
SPINE_LINKS: Tuple[str, ...] = (
    *(
        link
        for _ in range(NUM_TRANSFORMER_BLOCKS)
        for link in ("attention-add", "ffn-add")
    ),
    "layer-norm",
    "readout",
)

BRANCHES: Tuple[BranchSpec, ...] = (
    *(
        branch
        for block_index in range(NUM_TRANSFORMER_BLOCKS)
        for branch in (
            BranchSpec(
                key=f"block_{block_index}_attention",
                label=f"Block {block_index + 1} causal multi-head attention",
                reads=(
                    "embedding"
                    if block_index == 0
                    else block_node_key(block_index - 1, "ffn_residual")
                ),
                adds_before=block_node_key(block_index, "attention_residual"),
                path=(
                    block_node_key(block_index, "attention_input_norm"),
                    block_node_key(block_index, "attention_update"),
                ),
                observables=(
                    block_node_key(block_index, "attention_pattern"),
                ),
                kind="attention",
                block_index=block_index,
                side="above",
            ),
            BranchSpec(
                key=f"block_{block_index}_ffn",
                label=f"Block {block_index + 1} feed-forward network",
                reads=block_node_key(block_index, "attention_residual"),
                adds_before=block_node_key(block_index, "ffn_residual"),
                path=(
                    block_node_key(block_index, "ffn_input_norm"),
                    block_node_key(block_index, "ffn_hidden"),
                    block_node_key(block_index, "ffn_update"),
                ),
                kind="ffn",
                block_index=block_index,
                side="below",
            ),
        )
    ),
)

# The two decomposed embedding components that sum to the stream input.
EMBEDDING_COMPONENTS: Tuple[str, ...] = ("token_embeddings", "position_embeddings")

FAMILY_NODES: Dict[str, Tuple[str, ...]] = {
    node.family: tuple(
        candidate.key
        for candidate in STREAM_NODES
        if candidate.family == node.family
    )
    for node in STREAM_NODES
}


@dataclass(frozen=True)
class CapturedRun:
    locations: Dict[str, np.ndarray]
    probabilities: np.ndarray
    token_count: int


class AblationError(ValueError):
    pass


@dataclass(frozen=True)
class AblationSpec:
    node_key: str  # The node to ablate.
    dims: Tuple[int, ...]  # Dimensions to ablate.
    mode: str  # "zero" or "mean".
    scope: str  # "token" or "all".
    position: int | None = None  # Required for token-scoped ablation.

    def __post_init__(self) -> None:
        if self.node_key not in ABLATABLE_NODES:
            raise AblationError(
                f"Node cannot be ablated: {self.node_key}"
            )
        try:
            supplied_dims = tuple(self.dims)
        except TypeError as error:
            raise AblationError(
                "Ablation dimensions must be a sequence of integers"
            ) from error
        if not supplied_dims:
            raise AblationError("At least one ablation dimension is required")
        for dim in supplied_dims:
            if isinstance(dim, bool) or not isinstance(dim, int):
                raise AblationError("Ablation dimensions must be integers")
            if dim < 0:
                raise AblationError(
                    "Ablation dimensions must be non-negative"
                )
        object.__setattr__(self, "dims", tuple(sorted(set(supplied_dims))))
        if self.mode not in ABLATION_MODES:
            raise AblationError(
                f"Ablation mode must be one of: {', '.join(ABLATION_MODES)}"
            )
        if self.scope not in ABLATION_SCOPES:
            raise AblationError(
                f"Ablation scope must be one of: {', '.join(ABLATION_SCOPES)}"
            )
        if self.scope == "token":
            if self.position is None:
                raise AblationError(
                    "A token-scoped ablation requires a token position"
                )
            if (
                isinstance(self.position, bool)
                or not isinstance(self.position, int)
                or self.position < 0
            ):
                raise AblationError(
                    "Ablation token position must be a non-negative integer"
                )
        elif self.position is not None:
            raise AblationError(
                "An all-token ablation must not include a token position"
            )


def node_spec(key: str) -> StreamNode:
    try:
        return NODE_BY_KEY[key]
    except KeyError:
        raise InspectionError(f"Unknown stream node: {key}") from None


def node_width(node_key: str, config) -> int:
    if node_key not in ABLATABLE_NODES:
        raise AblationError(f"Node cannot be ablated: {node_key}")
    return (
        config.feed_forward_dim
        if node_spec(node_key).width_source == "ffn"
        else config.embedding_dim
    )


def ablation_node_options(config) -> list[dict]:
    return [
        {
            "key": key,
            "label": node_spec(key).label,
            "kind": node_spec(key).kind,
            "width": node_width(key, config),
        }
        for key in ABLATABLE_NODES
    ]


def _validate_ablation(
    checkpoint: LoadedCheckpoint,
    ablation: AblationSpec,
    token_count: int,
) -> None:
    width = node_width(ablation.node_key, checkpoint.config)
    invalid_dims = [dim for dim in ablation.dims if dim >= width]
    if invalid_dims:
        raise AblationError(
            f"Ablation dimension(s) {', '.join(map(str, invalid_dims))} are "
            f"outside the width of {ablation.node_key} ({width})"
        )
    if ablation.scope == "token":
        if ablation.position >= token_count:
            raise AblationError(
                f"Ablation token position {ablation.position} is outside "
                f"the prompt ({token_count} token(s))"
            )
        if ablation.mode == "mean" and token_count == 1:
            raise AblationError(
                "Mean ablation at one token requires another token for its "
                "leave-one-out baseline"
            )


def ablation_replacement_values(
    values: np.ndarray,
    ablation: AblationSpec,
) -> list[float]:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise AblationError("Ablation values must be two-dimensional")
    invalid_dims = [dim for dim in ablation.dims if dim >= matrix.shape[1]]
    if invalid_dims:
        raise AblationError(
            f"Ablation dimension(s) {', '.join(map(str, invalid_dims))} are "
            f"outside the tensor width ({matrix.shape[1]})"
        )
    if ablation.mode == "zero":
        return [0.0 for _ in ablation.dims]
    columns = matrix[:, list(ablation.dims)]
    if ablation.scope == "all":
        replacements = np.mean(columns, axis=0)
        return [float(value) for value in replacements]
    if ablation.position >= matrix.shape[0]:
        raise AblationError(
            f"Ablation token position {ablation.position} is outside the "
            f"tensor ({matrix.shape[0]} token(s))"
        )
    remaining = np.delete(columns, ablation.position, axis=0)
    if remaining.size == 0:
        raise AblationError(
            "Mean ablation at one token requires another token for its "
            "leave-one-out baseline"
        )
    replacements = np.mean(remaining, axis=0)
    return [float(value) for value in replacements]


def _apply_ablation(
    tensor: tf.Tensor,
    ablation: AblationSpec,
    token_count: int,
) -> tf.Tensor:
    matrix = tf.squeeze(tensor, axis=0)
    # Convert the requested feature dimensions into TensorFlow gather indices.
    dim_indices = tf.constant(ablation.dims, dtype=tf.int32)
    columns = tf.gather(matrix, dim_indices, axis=1)
    if ablation.mode == "zero":
        replacement_values = tf.zeros(
            [len(ablation.dims)],
            dtype=matrix.dtype,
        )
    elif ablation.scope == "all":
        replacement_values = tf.reduce_mean(columns, axis=0)
    else:
        replacement_values = (
            tf.reduce_sum(columns, axis=0) - columns[ablation.position]
        ) / tf.cast(token_count - 1, matrix.dtype)

    row_mask = (
        tf.ones([token_count], dtype=tf.bool)
        if ablation.scope == "all"
        else tf.equal(
            tf.range(token_count),
            tf.cast(ablation.position, tf.int32),
        )
    )
    column_mask = tf.reduce_any(
        tf.equal(
            tf.range(tf.shape(matrix)[1])[:, None],
            dim_indices[None, :],
        ),
        axis=1,
    )
    cell_mask = tf.logical_and(row_mask[:, None], column_mask[None, :])
    replacement_row = tf.reduce_sum(
        replacement_values[:, None]
        * tf.cast(
            tf.equal(
                dim_indices[:, None],
                tf.range(tf.shape(matrix)[1])[None, :],
            ),
            matrix.dtype,
        ),
        axis=0,
    )
    replacement_matrix = tf.broadcast_to(
        replacement_row[None, :],
        tf.shape(matrix),
    )
    return tf.expand_dims(
        tf.where(cell_mask, replacement_matrix, matrix),
        axis=0,
    )


def family_keys(family: str) -> Tuple[str, ...]:
    return FAMILY_NODES.get(family, (key for key in ()))


def capture_locations(
    checkpoint: LoadedCheckpoint,
    token_ids,
    ablation: AblationSpec | None = None,
) -> CapturedRun:
    """Run the loaded model once and return every internal tensor as a
    [sequence, width] NumPy array, all with dropout disabled. The attention
    pattern is stored as a [sequence, sequence] mean over heads."""
    if not isinstance(checkpoint, LoadedCheckpoint):
        raise TypeError("checkpoint must be a LoadedCheckpoint")

    ids = tf.convert_to_tensor(token_ids, dtype=tf.int32)
    if ids.shape.rank != 1:
        raise InspectionError("token ids must be a flat sequence")

    token_count = int(tf.size(ids))
    if token_count == 0:
        raise InspectionError("Prompt does not contain any tokens.")
    if token_count > checkpoint.config.max_len:
        raise InspectionError(
            f"Prompt has {token_count} tokens; the model accepts at most "
            f"{checkpoint.config.max_len}."
        )
    if ablation is not None:
        if not isinstance(ablation, AblationSpec):
            raise TypeError("ablation must be an AblationSpec")
        _validate_ablation(checkpoint, ablation, token_count)

    token_ids = ids[None, :]
    model = checkpoint.model
    embedding_layer = model.get_layer("token_and_position_embedding")
    token_embeddings = embedding_layer.token_emb(token_ids)
    positions = tf.range(token_count)[None, :]
    position_embeddings = embedding_layer.pos_emb(positions)
    embeddings = token_embeddings + position_embeddings
    ablation_node = node_spec(ablation.node_key) if ablation is not None else None
    if ablation is not None:
        if ablation.node_key == "embedding":
            embeddings = _apply_ablation(embeddings, ablation, token_count)
    locations = {
        "token_embeddings": tf.squeeze(token_embeddings, axis=0),
        "position_embeddings": tf.squeeze(position_embeddings, axis=0),
        "embedding": tf.squeeze(embeddings, axis=0),
    }
    block_output = embeddings
    for block_index in range(checkpoint.config.num_blocks):
        interventions = None
        if (
            ablation is not None
            and ablation_node.block_index == block_index
            and ablation_node.stage is not None
        ):
            interventions = {
                ablation_node.stage: lambda tensor: _apply_ablation(
                    tensor,
                    ablation,
                    token_count,
                )
            }
        steps, attention_scores = model.get_layer(
            f"transformer_block_{block_index}"
        ).call_steps(
            block_output,
            training=False,
            interventions=interventions,
        )
        pattern = tf.reduce_mean(attention_scores, axis=1)
        locations[block_node_key(block_index, "attention_pattern")] = tf.squeeze(
            pattern, axis=0
        )
        locations.update(
            {
                block_node_key(block_index, key): tf.squeeze(tensor, axis=0)
                for key, tensor in steps.items()
            }
        )
        block_output = steps["ffn_residual"]

    output_norm = model.get_layer("final_output_layer_norm")(block_output)
    if ablation is not None and ablation.node_key == "output_norm":
        output_norm = _apply_ablation(output_norm, ablation, token_count)
    probabilities = model.get_layer("token_probabilities")(output_norm)

    locations["output_norm"] = tf.squeeze(output_norm, axis=0)
    return CapturedRun(
        locations={key: value.numpy() for key, value in locations.items()},
        probabilities=tf.squeeze(probabilities, axis=0).numpy(),
        token_count=token_count,
    )
