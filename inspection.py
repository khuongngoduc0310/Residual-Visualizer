from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from checkpoint import LoadedCheckpoint


class InspectionError(ValueError):
    pass


DEFAULT_NODE_KEY = "output_norm"

# Kinds: component, stream, update, ln, hidden, pattern, readout.
# Families share a display color scale across their nodes:
#   components {token, position embeddings}
#   stream_raw {input, after-attention, after-FFN}
#   updates    {attention update, FFN update}
#   stream_norm{attention_norm, output_norm}
#   hidden     {FFN hidden}   (single, non-negative sequential scale)
#   pattern    {attention pattern} (fixed 0..1)
#   readout    {probabilities / entropy}
COMPONENTS = "components"
STREAM_RAW = "stream_raw"
UPDATES = "updates"
STREAM_NORM = "stream_norm"
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


STREAM_NODES: Tuple[StreamNode, ...] = (
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
            "the transformer block. The attention heads read this value."
        ),
    ),
    StreamNode(
        key="attention_pattern",
        label="Causal attention pattern",
        kind="pattern",
        family=PATTERN,
        explanation=(
            "Attention weights averaged over the heads, averaged over heads "
            "when there are several. Row (query token) sums to about one; "
            "columns to the right of each row are masked by causality."
        ),
        feature_axis=False,
    ),
    StreamNode(
        key="attention_update",
        label="Attention output \u2192 residual",
        kind="update",
        family=UPDATES,
        explanation=(
            "The value the causal attention heads write into the residual "
            "stream. Adding this to the stream input produces the residual "
            "stream after attention."
        ),
    ),
    StreamNode(
        key="attention_residual",
        label="Residual stream \u00b7 after attention",
        kind="stream",
        family=STREAM_RAW,
        explanation=(
            "The residual stream after the attention update is added back to "
            "the input embeddings. Feature magnitudes here combine the "
            "embeddings with what attention just wrote."
        ),
    ),
    StreamNode(
        key="attention_norm",
        label="Layer norm \u00b7 after attention",
        kind="ln",
        family=STREAM_NORM,
        normalized=True,
        explanation=(
            "Layer normalization sits on the residual line and rescales every "
            "token to unit variance, so magnitude comparisons across tokens "
            "here are not meaningful."
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
    ),
    StreamNode(
        key="ffn_update",
        label="FFN output \u2192 residual",
        kind="update",
        family=UPDATES,
        explanation=(
            "The value the feed-forward network writes into the residual "
            "stream. Adding this to the normalized attention output produces "
            "the residual stream after the FFN."
        ),
    ),
    StreamNode(
        key="ffn_residual",
        label="Residual stream \u00b7 after FFN",
        kind="stream",
        family=STREAM_RAW,
        explanation=(
            "The residual stream after the FFN update is added to the "
            "normalized attention output. This is the value fed to the final "
            "layer norm."
        ),
    ),
    StreamNode(
        key="output_norm",
        label="Layer norm \u00b7 block output",
        kind="ln",
        family=STREAM_NORM,
        normalized=True,
        explanation=(
            "The final layer-normalized block output. It is what the "
            "vocabulary projection reads to predict the next token."
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
    ),
)

NODE_BY_KEY: Dict[str, StreamNode] = {node.key: node for node in STREAM_NODES}
TRACE_ORDER: Tuple[str, ...] = tuple(node.key for node in STREAM_NODES)
# Every tensor actually captured in the single analysis run. The readout node
# is virtual: it is derived from the stored probabilities matrix.
CAPTURED_KEYS: Tuple[str, ...] = tuple(
    key for key in TRACE_ORDER if key != "readout"
)

# Central line through the model, in flow order.
SPINE_STATES: Tuple[str, ...] = (
    "embedding",
    "attention_residual",
    "attention_norm",
    "ffn_residual",
    "output_norm",
)

# Describes the connector between successive spine states and where it starts
# from line value (used for the branch wiring too).
@dataclass(frozen=True)
class SpineLink:
    label: str
    branch_key: str | None


# Links between consecutive SPINE_STATES plus the readout at the end.
#   embedding -> attention_residual  : attention add junction
#   attention_residual -> attention_norm : layer norm (in line)
#   attention_norm -> ffn_residual   : ffn add junction
#   ffn_residual -> output_norm      : layer norm (in line)
#   output_norm -> readout           : projection + softmax
SPINE_LINKS: Tuple[str, ...] = (
    "attention-add",
    "layer-norm",
    "ffn-add",
    "layer-norm",
    "readout",
)

BRANCHES: Tuple[Dict[str, object], ...] = (
    {
        "key": "attention",
        "label": "Causal multi-head attention",
        "reads": "embedding",
        "adds_before": "attention_residual",
        "nodes": ("attention_pattern", "attention_update"),
    },
    {
        "key": "ffn",
        "label": "Feed-forward network",
        "reads": "attention_norm",
        "adds_before": "ffn_residual",
        "nodes": ("ffn_hidden", "ffn_update"),
    },
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


def node_spec(key: str) -> StreamNode:
    try:
        return NODE_BY_KEY[key]
    except KeyError:
        raise InspectionError(f"Unknown stream node: {key}") from None


def family_keys(family: str) -> Tuple[str, ...]:
    return FAMILY_NODES.get(family, (key for key in ()))


def capture_locations(checkpoint: LoadedCheckpoint, token_ids) -> CapturedRun:
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

    token_ids = ids[None, :]
    model = checkpoint.model
    embedding_layer = model.get_layer("token_and_position_embedding")
    token_embeddings = embedding_layer.token_emb(token_ids)
    positions = tf.range(token_count)[None, :]
    position_embeddings = embedding_layer.pos_emb(positions)
    embeddings = token_embeddings + position_embeddings
    steps, attention_scores = model.get_layer(
        "transformer_block"
    ).call_steps(embeddings, training=False)
    block_output = steps["output_norm"]
    probabilities = model.get_layer("token_probabilities")(block_output)

    pattern = tf.reduce_mean(attention_scores, axis=1)
    locations = {
        "token_embeddings": tf.squeeze(token_embeddings, axis=0),
        "position_embeddings": tf.squeeze(position_embeddings, axis=0),
        "embedding": tf.squeeze(embeddings, axis=0),
        "attention_pattern": tf.squeeze(pattern, axis=0),
        **{
            key: tf.squeeze(tensor, axis=0)
            for key, tensor in steps.items()
        },
    }
    return CapturedRun(
        locations={key: value.numpy() for key, value in locations.items()},
        probabilities=tf.squeeze(probabilities, axis=0).numpy(),
        token_count=token_count,
    )
